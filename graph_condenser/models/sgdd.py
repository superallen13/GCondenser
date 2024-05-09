from graph_condenser.models.condenser import Condenser
from graph_condenser.models.gcond import match_loss

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def get_adj_norm(g):
    if g.num_nodes() > 5000:
        edge_index, edge_weight = gcn_norm(g.edge_index)
        random_nodes = torch.randperm(g.num_nodes())[:5000].to(g.device)
        adj = torch.zeros((5000, 5000), device=edge_weight.device)
        src, dst = g.edges()

        mask = torch.isin(src, random_nodes) & torch.isin(dst, random_nodes)
        filtered_src = src[mask]
        filtered_dst = dst[mask]
        filtered_weights = edge_weight[mask]

        node_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(random_nodes.tolist())
        }
        mapped_src = torch.tensor([node_mapping[x.item()] for x in filtered_src])
        mapped_dst = torch.tensor([node_mapping[x.item()] for x in filtered_dst])

        adj[mapped_src, mapped_dst] = filtered_weights
    else:
        edge_index, edge_weight = gcn_norm(g.edge_index)
        adj = torch.zeros((g.num_nodes(), g.num_nodes()), device=edge_weight.device)
        src, dst = g.edges()
        adj[src, dst] = edge_weight
    return adj


def feature_smoothing(adj, X):
    adj = (adj.t() + adj) / 2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv + 1e-8
    r_inv = r_inv.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag(r_inv)

    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)

    return loss_smooth_feat


def regularization(adj, x, eig_real=None):
    loss = 0
    loss += feature_smoothing(adj, x)
    return loss


class SGDD(Condenser):
    def __init__(
        self,
        dataset,
        observe_mode: str,
        budget: int,
        label_distribution: str,
        init_method: str,
        val_mode: str,
        gnn_val: torch.nn.Module,
        lr_val: float,
        wd_val: float,
        gnn_test: torch.nn.Module,
        lr_test: float,
        wd_test: float,
        structure_generator: torch.nn.Module,
        opt_feat: torch.optim.Optimizer,
        gnn: torch.nn.Module,
        lr_model: float,
        wd_model: float,
        opt_adj: torch.optim.Optimizer,
        loop_outer: int,
        loop_inner: int,
        update_epoch_adj: int,
        update_epoch_feat: int,
        dis_metric: str,
        beta: float,
        opt_scale: float,
    ):
        super().__init__(
            dataset,
            observe_mode,
            budget,
            label_distribution,
            init_method,
            val_mode,
            gnn_val,
            lr_val,
            wd_val,
            gnn_test,
            lr_test,
            wd_test,
        )
        # change the condensed graph to the fully connected graph.
        i, j = torch.meshgrid(torch.arange(budget), torch.arange(budget), indexing="ij")
        self.g_cond.edge_index = torch.stack([i.flatten(), j.flatten()])
        assert self.g_cond.edge_index.size(1) == budget**2
        self.structure_generator = structure_generator(
            nnodes=self.feat_cond.size(0), nfeat=self.feat_cond.size(1)
        )
        self.gnn = gnn

    def training_step(self, data, batch_idx):
        g = data
        feat = g.x
        label = g.y

        g_cond = self.g_cond.to(self.device)
        feat_cond = self.feat_cond
        label_cond = g_cond.y

        gnn = self.gnn(self.num_node_features, self.num_classes)
        gnn.to(self.device)
        model_parameters = list(gnn.parameters())
        opt_model = torch.optim.Adam(model_parameters, lr=self.hparams.lr_model)
        gnn.train()

        class_mask_real = [(label == c) & g.train_mask for c in range(self.num_classes)]
        class_mask_cond = [(label_cond == c) for c in range(self.num_classes)]

        opt_feat, opt_adj = self.optimizers()

        losses = []
        for ol in range(self.hparams.loop_outer):
            adj_norm = get_adj_norm(g)
            adj_cond, edge_weight_cond, opt_loss = self.structure_generator(
                feat_cond, Lx=adj_norm
            )

            output = gnn(feat, g.edge_index)
            output_cond = gnn(g_cond, feat_cond, edge_weight_cond)

            loss = torch.tensor(0.0).to(self.device)
            for i, c in enumerate(range(self.num_classes)):
                loss_real = F.cross_entropy(
                    output[class_mask_real[i]], label[class_mask_real[i]]
                )
                gw_orig = torch.autograd.grad(
                    loss_real, model_parameters, retain_graph=True
                )
                gw_orig = list((_.detach().clone() for _ in gw_orig))

                loss_cond = F.cross_entropy(
                    output_cond[class_mask_cond[i]],
                    label_cond[class_mask_cond[i]],
                )
                gw_cond = torch.autograd.grad(
                    loss_cond, model_parameters, create_graph=True
                )

                loss += match_loss(gw_cond, gw_orig, dis_metric=self.hparams.dis_metric)

            if self.hparams.beta > 0:
                loss_reg = self.hparams.beta * regularization(
                    adj_cond,
                    F.one_hot(torch.LongTensor(label_cond.cpu()), self.num_classes)
                    .float()
                    .to(self.device),
                )
            else:
                loss_reg = torch.tensor(0)

            loss = loss + loss_reg

            losses.append(loss.item())
            # update sythetic graph
            opt_adj.zero_grad()
            opt_feat.zero_grad()
            self.manual_backward(loss)
            if (
                self.current_epoch
                % (self.hparams.update_epoch_adj + self.hparams.update_epoch_feat)
                < self.hparams.update_epoch_adj
            ):
                loss = loss + self.hparams.opt_scale * opt_loss
                opt_adj.step()
            else:
                opt_feat.step()

            if self.hparams.loop_inner > 0 and ol != self.hparams.loop_outer - 1:
                feat_cond_inner = feat_cond.detach()
                edge_weight_cond = self.structure_generator.inference(feat_cond_inner)
                for il in range(self.hparams.loop_inner):
                    opt_model.zero_grad()
                    output_cond_inner = gnn(g_cond, feat_cond_inner, edge_weight_cond)
                    loss_cond_inner = F.cross_entropy(output_cond_inner, label_cond)
                    loss_cond_inner.backward()
                    opt_model.step()
        loss_avg = sum(losses) / len(losses)
        self.log(
            "train/loss",
            loss_avg,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        opt_feat = self.hparams.opt_feat(params=[self.feat_cond])
        opt_adj = self.hparams.opt_adj(params=self.structure_generator.parameters())
        return opt_feat, opt_adj

from graph_condenser.models.condenser import Condenser

import torch
import torch.nn.functional as F


def match_loss(gw_syn, gw_real, dis_metric="ours"):
    if dis_metric == "ours":
        dis = torch.tensor(0.0).to(gw_syn[0].device)
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += torch.sum(
                1
                - torch.sum(gwr * gws, dim=-1)
                / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 1e-6)
            )

    elif dis_metric == "mse":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == "cos":
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
            torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 1e-6
        )

    else:
        exit("DC error: unknown distance function")

    return dis


class GCond(Condenser):
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
        if structure_generator is None:
            self.structure_generator = None
        else:
            # change the condensed graph to the fully connected graph.
            i, j = torch.meshgrid(
                torch.arange(budget), torch.arange(budget), indexing="ij"
            )
            self.g_cond.edge_index = torch.stack([i.flatten(), j.flatten()])
            assert self.g_cond.edge_index.size(1) == budget**2
            self.structure_generator = structure_generator(self.g_cond)
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

        if self.structure_generator is None:
            opt_feat = self.optimizers()
        else:
            opt_feat, opt_adj = self.optimizers()

        losses = []
        for ol in range(self.hparams.loop_outer):
            if self.structure_generator:
                edge_weight_cond = self.structure_generator(feat_cond)
            else:
                edge_weight_cond = None

            output = gnn(feat, g.edge_index)
            output_cond = gnn(feat_cond, g_cond.edge_index, edge_weight_cond)

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
            losses.append(loss.item())

            # update sythetic graph
            if self.structure_generator is not None:
                opt_adj.zero_grad()
            opt_feat.zero_grad()
            self.manual_backward(loss)
            if self.structure_generator is not None:
                if (
                    self.current_epoch
                    % (self.hparams.update_epoch_adj + self.hparams.update_epoch_feat)
                    < self.hparams.update_epoch_adj
                ):
                    opt_adj.step()
                else:
                    opt_feat.step()
            else:
                opt_feat.step()

            if self.hparams.loop_inner > 0 and ol != self.hparams.loop_outer - 1:
                feat_cond_inner = feat_cond.detach()
                if self.structure_generator:
                    edge_weight_cond = self.structure_generator.inference(
                        feat_cond_inner
                    )
                else:
                    edge_weight_cond = None
                for il in range(self.hparams.loop_inner):
                    opt_model.zero_grad()
                    output_cond_inner = gnn(
                        feat_cond_inner, g_cond.edge_index, edge_weight_cond
                    )
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
        if self.structure_generator:
            opt_adj = self.hparams.opt_adj(params=self.structure_generator.parameters())
            return opt_feat, opt_adj
        else:
            opt_adj = None
            return opt_feat

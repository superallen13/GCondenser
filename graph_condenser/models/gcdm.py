from graph_condenser.models.condenser import Condenser

import torch


def mean_emb(emb, y):
    classes = torch.unique(y)
    return torch.stack([torch.mean(emb[y == cls], 0) for cls in classes])


class GCDM(Condenser):
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

        gnn = self.gnn(self.num_node_features, self.num_classes).to(self.device)
        model_parameters = list(gnn.parameters())
        opt_model = torch.optim.Adam(model_parameters, lr=self.hparams.lr_model)
        gnn.train()

        if self.structure_generator is None:
            opt_feat = self.optimizers()
        else:
            opt_feat, opt_adj = self.optimizers()

        with torch.no_grad():
            emb_orig = gnn.encode(feat, g.edge_index)
            mean_emb_orig = mean_emb(emb_orig[g.train_mask], label[g.train_mask])[
                torch.unique(label_cond)
            ]

        losses = []
        for ol in range(self.hparams.loop_outer):
            # Improve efficiency by freezing the model parameters.
            for param in gnn.parameters():
                param.requires_grad_(False)

            if self.structure_generator:
                edge_weight_cond = self.structure_generator(feat_cond)
                self.g_cond.edge_weight = edge_weight_cond.detach().to("cpu")
            else:
                edge_weight_cond = None

            emb_cond = gnn.encode(feat_cond, g_cond.edge_index, edge_weight_cond)
            mean_emb_cond = mean_emb(emb_cond, label_cond)

            emb_diff = (mean_emb_cond - mean_emb_orig) ** 2
            loss = torch.sum(torch.mean(emb_diff, dim=1))
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

            # Enter the inner loop.
            if self.hparams.loop_inner > 0 and ol != self.hparams.loop_outer - 1:
                feat_cond_inner = feat_cond.detach()
                if self.structure_generator:
                    edge_weight_cond = self.structure_generator.inference(
                        feat_cond_inner
                    )
                else:
                    edge_weight_cond = None

                for param in gnn.parameters():
                    param.requires_grad_(True)


                for il in range(self.hparams.loop_inner):
                    emb_cond = gnn.encode(
                        feat_cond_inner, g_cond.edge_index, edge_weight_cond
                    )
                    mean_emb_cond = mean_emb(emb_cond, label_cond)

                    emb_diff = (mean_emb_cond - mean_emb_orig) ** 2
                    loss_syn_inner = torch.sum(torch.mean(emb_diff, dim=1))

                    opt_model.zero_grad()
                    loss_syn_inner.backward()
                    opt_model.step()

                    # Update embedding of the original grpah once the mode is updated.
                    with torch.no_grad():
                        emb_orig = gnn.encode(feat, g.edge_index)
                        mean_emb_orig = mean_emb(
                            emb_orig[g.train_mask],
                            label[g.train_mask],
                        )[torch.unique(label_cond)]

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

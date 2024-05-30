from typing import Union

from graph_condenser.models.condenser import Condenser
from graph_condenser.models.components.trajectories import TrajectoryBuffer
from graph_condenser.models.components.reparam_module import ReparamModule

import dgl
import numpy as np
import torch
import torch.nn.functional as F


class SFGC(Condenser):
    def __init__(
        self,
        g_orig,
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
        gnn: torch.nn.Module,
        lr_model: float,
        opt_feat: torch.optim.Optimizer,
        traj_buffer: TrajectoryBuffer,
        optim_lr: Union[str, None],
        lr_lr: float,
        rand_start: bool,
        start_epoch: int,
        expert_epochs: int,
        syn_steps: int,
        batch_training: bool = False,
    ):
        super().__init__(
            g_orig,
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
        self.gnn = gnn
        self.traj_buffer = traj_buffer
        self.traj_buffer.load()

    def training_step(self, data, batch_idx):
        g_cond = self.g_cond.to(self.device)
        feat_cond = self.feat_cond
        label_cond = g_cond.ndata["label"]

        model = ReparamModule(self.gnn(self.num_node_features, self.num_classes))
        model.train()
        num_params = sum([np.prod(p.size()) for p in (model.parameters())])

        self.traj_buffer.next()
        starting_params = self.traj_buffer.get(
            self.hparams.start_epoch, rand_start=self.hparams.rand_start
        )

        starting_params = torch.cat(
            [p.data.to(self.device).reshape(-1) for p in starting_params], 0
        )

        student_params = [
            torch.cat(
                [p.data.to(self.device).reshape(-1) for p in starting_params], 0
            ).requires_grad_(True)
        ]

        target_params = self.traj_buffer.get(
            self.hparams.start_epoch + self.hparams.expert_epochs
        )

        target_params = torch.cat(
            [p.data.to(self.device).reshape(-1) for p in target_params], 0
        )

        syn_lr = torch.tensor(self.hparams.lr_model).to(self.device)
        if self.hparams.optim_lr:
            syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
            if self.hparams.optim_lr.lower() == "adam":
                opt_lr = torch.optim.Adam([syn_lr], lr=self.hparams.lr_lr)
            elif self.hparams.optim_lr.lower() == "sgd":
                opt_lr = torch.optim.SGD([syn_lr], lr=self.hparams.lr_lr, momentum=0.5)

        for step in range(self.hparams.syn_steps):
            forward_params = student_params[-1]
            output_syn = model(g_cond, feat_cond, flat_param=forward_params)
            loss_syn = F.cross_entropy(output_syn, label_cond)
            grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)
            student_params.append(student_params[-1] - syn_lr * grad[0])
        param_loss = (
            torch.nn.functional.mse_loss(
                student_params[-1], target_params, reduction="sum"
            )
            / num_params
        )

        self.log(
            "train/loss",
            param_loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        opt_feat = self.optimizers()
        opt_feat.zero_grad()
        if self.hparams.optim_lr:
            opt_lr.zero_grad()
        self.manual_backward(param_loss)
        opt_feat.step()
        if self.hparams.optim_lr:
            opt_lr.step()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # pop the traj_buffer which is located on the "hyper_parameters"
        checkpoint["hyper_parameters"].pop("traj_buffer")

    def configure_optimizers(self):
        opt_feat = self.hparams.opt_feat(params=[self.feat_cond])
        return opt_feat

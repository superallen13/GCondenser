from typing import Union
import copy
import os

from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.condenser import Condenser
from graph_condenser.models.components.trajectories_cl import TrajectoryBuffer
from graph_condenser.models.components.reparam_module import ReparamModule
from graph_condenser.models.backbones.lightning_gnn import LightningGNN


import dgl
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
import torch.nn.functional as F


class GEOM(Condenser):
    def __init__(
        self,
        g,
        observe_mode: str,
        budget: int,
        label_distribution: str,
        init_method: str,
        validator: torch.nn.Module,
        lr_val: float,
        wd_val: float,
        tester: torch.nn.Module,
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
        expanding_window: bool,
        window_left_epoch: int,
        window_right_epoch: int,
        soft_label: bool,
        lr_label: float,
        alpha: float,
    ):
        super().__init__(
            g,
            observe_mode,
            budget,
            label_distribution,
            init_method,
            validator,
            lr_val,
            wd_val,
            tester,
            lr_test,
            wd_test,
        )
        self.gnn = gnn
        self.traj_buffer = traj_buffer
        self.traj_buffer.load()

        self.syn_lr = torch.tensor(lr_model)
        if optim_lr:
            self.syn_lr = torch.tensor(lr_model).requires_grad_(True)
            if optim_lr.lower() == "adam":
                self.opt_lr = torch.optim.Adam([self.syn_lr], lr=lr_lr)
            elif optim_lr.lower() == "sgd":
                self.opt_lr = torch.optim.SGD([self.syn_lr], lr=lr_lr, momentum=0.5)

        if soft_label:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_ = ReparamModule(self.gnn(self.num_node_features, self.num_classes))
            model_.eval()
            self.traj_buffer.next()
            params = self.traj_buffer.expert_trajectory[-1]
            params = torch.cat([p.data.to(device).reshape(-1) for p in params], 0)

            output = model_(
                self.g_cond.to(device), self.feat_cond.to(device), flat_param=params
            )
            
            self.label_cond = copy.deepcopy(F.log_softmax(output.detach().cpu(), dim=1)).requires_grad_(True)
            self.opt_label = torch.optim.SGD(
                [self.label_cond], lr=lr_label, momentum=0.9
            )

    def training_step(self, data, batch_idx):
        g_cond = self.g_cond.to(self.device)
        feat_cond = self.feat_cond
        syn_lr = self.syn_lr.to(self.device)
        if self.hparams.soft_label:
            label_cond = self.label_cond.to(self.device)
        else:
            label_cond = g_cond.ndata["label"]

        self.traj_buffer.next()  # randomly select a trajectory
        # get the current upper bound of the starting epoch
        if self.hparams.expanding_window:
            if self.current_epoch < self.hparams.window_left_epoch:
                start_epoch = self.hparams.window_left_epoch
            elif (
                self.current_epoch >= self.hparams.window_left_epoch
                and self.current_epoch < self.hparams.window_right_epoch
            ):
                start_epoch = self.current_epoch
            else:
                start_epoch = self.hparams.window_right_epoch
        else:
            start_epoch = self.hparams.start_epoch

        self.log(
            "train/start_epoch",
            start_epoch,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        find_epoch, starting_params = self.traj_buffer.get(
            start_epoch, rand_start=self.hparams.rand_start
        )
        starting_params = torch.cat(
            [p.data.to(self.device).reshape(-1) for p in starting_params], 0
        )
        student_params = [
            torch.cat(
                [p.data.to(self.device).reshape(-1) for p in starting_params], 0
            ).requires_grad_(True)
        ]
        _, target_params = self.traj_buffer.get(find_epoch + self.hparams.expert_epochs)
        target_params = torch.cat(
            [p.data.to(self.device).reshape(-1) for p in target_params], 0
        )

        model = ReparamModule(self.gnn(self.num_node_features, self.num_classes))
        model.train()
        num_params = sum([np.prod(p.size()) for p in (model.parameters())])

        for step in range(self.hparams.syn_steps):
            forward_params = student_params[-1]
            output_syn = model(g_cond, feat_cond, flat_param=forward_params)
            if self.hparams.soft_label:
                loss_syn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                    F.log_softmax(output_syn, dim=1), label_cond
                )
            else:
                loss_syn = F.cross_entropy(output_syn, label_cond)
            grad = torch.autograd.grad(loss_syn, student_params[-1], create_graph=True)
            student_params.append(student_params[-1] - syn_lr * grad[0])
        param_loss = torch.norm(student_params[-1] - target_params, 2)
        param_dist = torch.norm(starting_params - target_params, 2)

        if self.hparams.soft_label:
            params_optimal = self.traj_buffer.expert_trajectory[-1]
            params_optimal = torch.cat(
                [p.data.to(self.device).reshape(-1) for p in params_optimal], 0
            )
            model_optimal = ReparamModule(
                self.gnn(self.num_node_features, self.num_classes)
            )
            model_optimal.eval()
            output_optimal = model_optimal(g_cond, feat_cond, flat_param=params_optimal)
            loss_kee = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                F.log_softmax(output_optimal, dim=1), label_cond
            )
            total_loss = param_loss / param_dist + self.hparams.alpha * loss_kee
            self.log(
                "train/loss_kee",
                self.hparams.alpha * loss_kee,
                batch_size=1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            total_loss = param_loss / param_dist

        self.log(
            "train/param_loss",
            param_loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train/param_dist",
            param_dist,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train/total_loss",
            param_loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        opt_feat = self.optimizers()
        opt_feat.zero_grad()
        if self.hparams.optim_lr:
            self.opt_lr.zero_grad()
            self.log(
                "train/syn_lr",
                syn_lr,
                batch_size=1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        if self.hparams.soft_label:
            self.opt_label.zero_grad()
        self.manual_backward(total_loss)
        opt_feat.step()
        if self.hparams.optim_lr:
            self.opt_lr.step()
        if self.hparams.soft_label:
            self.opt_label.step()

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # pop the traj_buffer which is located on the "hyper_parameters"
        checkpoint["hyper_parameters"].pop("traj_buffer")

    def configure_optimizers(self):
        opt_feat = self.hparams.opt_feat(params=[self.feat_cond])
        return opt_feat

    def validation_step(self, data, batch_idx):
        if self.hparams.soft_label:
            g_cond = self.g_cond.to(self.device)
            g_cond.ndata["feat"] = self.feat_cond.data.clone()
            g_cond.ndata["soft_label"] = self.label_cond.data.clone().to(self.device)

            gnn_val = self.validator(self.num_node_features, self.num_classes)
            datamodule = CondensedGraphDataModule(
                g_cond, data, self.hparams.observe_mode
            )
            model = LightningGNN(gnn_val, self.hparams.lr_val, self.hparams.wd_val)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=200,
                log_every_n_steps=1,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
            )
            trainer.fit(model, datamodule)
            self.log(
                "val/acc",
                model.val_acc_best.compute(),
                batch_size=1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            self.val_acc_best(model.val_acc_best.compute())
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                "val/acc_best",
                self.val_acc_best.compute(),
                batch_size=1,
                sync_dist=True,
                prog_bar=True,
            )

            # make sure the memory used in the validation step is released
            del datamodule, gnn_val, model, trainer
            torch.cuda.empty_cache()
            pl.utilities.memory.garbage_collection_cuda()
        else:
            super().validation_step(data, batch_idx)

    def test_step(self, data, batch_idx):
        if self.hparams.soft_label:
            g_cond = self.g_cond.to(self.device)
            g_cond.ndata["feat"] = self.feat_cond.data.clone()
            g_cond.ndata["soft_label"] = self.label_cond.data.clone().to(self.device)

            datamodule = CondensedGraphDataModule(
                g_cond, data, self.hparams.observe_mode
            )
            torch.set_grad_enabled(True)  # enable gradients for the test step

            results = []
            table_cols = ["test_acc_mean", "test_acc_std"]
            for i in range(5):
                gnn_test = self.tester(self.num_node_features, self.num_classes)
                model = LightningGNN(
                    gnn_test, self.hparams.lr_test, self.hparams.wd_test
                )
                checkpoint = ModelCheckpoint(
                    dirpath=self.test_ckpt_dir,
                    monitor="val_acc",
                    save_top_k=1,
                    mode="max",
                )
                trainer = Trainer(
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    devices=1,
                    max_epochs=100,
                    log_every_n_steps=1,
                    callbacks=[checkpoint],
                    enable_model_summary=False,
                    enable_progress_bar=False,
                    logger=False,
                )
                trainer.fit(model, datamodule)
                ckpt_path = checkpoint.best_model_path
                test_acc = trainer.test(
                    model=model,
                    datamodule=datamodule,
                    ckpt_path=ckpt_path,
                    verbose=True,
                )
                results.append(test_acc[0]["test_acc"])
                table_cols.append(f"test_acc_{i}")
                os.remove(ckpt_path)

            acc_mean = sum(results) / len(results)
            acc_std = torch.std(torch.tensor(results)).item()
            table_data = [acc_mean, acc_std] + results

            if self.logger is not None:
                # assume the logger is WandbLogger
                self.logger.log_table(
                    key="test/acc_table", columns=table_cols, data=[table_data]
                )

            self.log(
                "test/acc_mean",
                acc_mean,
                batch_size=1,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                "test/acc_std",
                acc_std,
                batch_size=1,
                sync_dist=True,
                prog_bar=True,
            )

            return acc_mean, acc_std
        else:
            super().test_step(data, batch_idx)

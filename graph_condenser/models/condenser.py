import os

from graph_condenser.data.utils import get_npc, get_g_cond
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.backbones.lightning_gnn import LightningGNN
from graph_condenser.models.validators.sgc import SGCEvaluator
from graph_condenser.models.validators.lightning_evaluator import LightningEvaluator
from graph_condenser.models.components.gntk import GNTK


import dgl
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from hydra.core.hydra_config import HydraConfig
from torchmetrics import MaxMetric


class Condenser(LightningModule):
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
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["g", "validator", "tester"])
        self.automatic_optimization = False
        self.num_node_features = g.ndata["feat"].shape[1]
        self.num_classes = g.num_classes

        # Initilaise the condensed graph.
        self.npc = get_npc(g, budget, label_distribution)
        self.g_cond = get_g_cond(g, observe_mode, self.npc, init_method)
        self.feat_cond = self.g_cond.ndata["feat"]

        # instantiate the GNNs in the validation and test steps
        self.validator = validator
        self.tester = tester

        self.val_acc_best = MaxMetric()

        # create a directory to store the checkpoint for the test step
        output_dir = HydraConfig.get().runtime.output_dir
        self.test_ckpt_dir = os.path.join(output_dir, "test_step_ckpt")
        if not os.path.exists(self.test_ckpt_dir):
            os.makedirs(self.test_ckpt_dir)

    def forward(self):
        raise NotImplementedError

    def on_train_start(self):
        self.val_acc_best.reset()

    def training_step(self, data, batch_idx):
        raise NotImplementedError

    def validation_step(self, data, batch_idx):
        g_cond = self.g_cond.to(self.device)
        g_cond.ndata["feat"] = self.feat_cond.data.clone()
        if getattr(self, "structure_generator", None) is not None:
            g_cond.edata["weight"] = self.structure_generator.inference(
                g_cond.ndata["feat"]
            )

        if self.validator == "gntk":
            gntk = GNTK(num_layers=3, num_mlp_layers=2, scale="degree")

            data_val = dgl.node_subgraph(data, data.ndata["val_mask"])
            num_nodes = data_val.num_nodes()

            if num_nodes > 5000:
                num_per_class = [0] * self.num_classes
                for i in range(self.num_classes):
                    num_per_class[i] = (data_val.ndata["label"] == i).sum().item()
                # let the sum of the num_per_class be at most 5000
                # each class at least 1
                # keep the distribution of the number of nodes per class
                while sum(num_per_class) > 5000:
                    for i in range(self.num_classes):
                        if num_per_class[i] > 1:
                            num_per_class[i] -= 1
                            if sum(num_per_class) <= 5000:
                                break
                random_nodes = []
                # randomly choose the nodes with the corresponding label
                for i in range(self.num_classes):
                    nodes = torch.where(data_val.ndata["label"] == i)[0]
                    random_nodes.append(
                        nodes[torch.randperm(len(nodes))[: num_per_class[i]]]
                    )
                random_nodes = torch.cat(random_nodes)
                random_val_g = dgl.node_subgraph(data, random_nodes)
                feat_val = random_val_g.ndata["feat"]
                adj_val = torch.zeros(5000, 5000).to(self.device)
                src, dst = random_val_g.edges()
                adj_val[src, dst] = 1
                y_target = F.one_hot(
                    random_val_g.ndata["label"], self.num_classes
                ).float()
            else:
                feat_val = data_val.ndata["feat"]
                adj_val = torch.zeros(num_nodes, num_nodes).to(self.device)
                src, dst = data_val.edges()
                adj_val[src, dst] = 1
                y_target = F.one_hot(data_val.ndata["label"], self.num_classes).float()

            # creat the adj of the condensed graph
            edge_weight = g_cond.edata["weight"] if "weight" in g_cond.edata else None
            if edge_weight is not None:
                adj_cond = torch.zeros(sum(self.npc), sum(self.npc)).to(self.device)
                src, dst = g_cond.edges()
                adj_cond[src, dst] = edge_weight
            else:
                adj_cond = torch.eye(sum(self.npc)).to(self.device)

            y_cond = F.one_hot(g_cond.ndata["label"], self.num_classes).float()

            pred, K_SS = gntk(adj_cond, g_cond.ndata["feat"], y_cond, adj_val, feat_val)

            mse_loss = 0.5 * F.mse_loss(pred, y_target)
            acc = (pred.argmax(dim=1) == y_target.argmax(dim=1)).float().mean()
            self.log(
                "val/acc",
                acc,
                batch_size=1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.val_acc_best(acc)
            self.log(
                "val/acc_best",
                self.val_acc_best.compute(),
                batch_size=1,
                sync_dist=True,
                prog_bar=True,
            )
        else:
            gnn_val = self.validator(self.num_node_features, self.num_classes)
            datamodule = CondensedGraphDataModule(
                g_cond, data, self.hparams.observe_mode
            )
            if isinstance(gnn_val, SGCEvaluator):
                model = LightningEvaluator(
                    gnn_val, self.hparams.lr_val, self.hparams.wd_val
                )
            else:
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

    def on_save_checkpoint(self, checkpoint):
        # save the g_cond
        g_cond = self.g_cond
        feat_cond = self.feat_cond
        g_cond.ndata["feat"] = feat_cond.to("cpu").data.clone()
        if getattr(self, "structure_generator", None) is not None:
            edge_weight = self.structure_generator.inference(feat_cond)
            g_cond.edata["weight"] = edge_weight.to("cpu")
        checkpoint["g_cond"] = g_cond

    def test_step(self, data, batch_idx):
        g_cond = self.g_cond.to(self.device)
        g_cond.ndata["feat"] = self.feat_cond.data.clone()
        if getattr(self, "structure_generator", None) is not None:
            g_cond.edata["weight"] = self.structure_generator.inference(
                g_cond.ndata["feat"]
            )

        datamodule = CondensedGraphDataModule(g_cond, data, self.hparams.observe_mode)
        torch.set_grad_enabled(True)  # enable gradients for the test step

        results = []
        table_cols = ["test_acc_mean", "test_acc_std"]
        for i in range(5):
            gnn_test = self.tester(self.num_node_features, self.num_classes)
            model = LightningGNN(gnn_test, self.hparams.lr_test, self.hparams.wd_test)
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
                model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
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

    def on_test_end(self):
        # remove ckpt dir of the test setep.
        for file in os.listdir(self.test_ckpt_dir):
            os.remove(os.path.join(self.test_ckpt_dir, file))
        os.rmdir(self.test_ckpt_dir)

    def configure_optimizers(self):
        raise NotImplementedError

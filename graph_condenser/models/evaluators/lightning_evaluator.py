import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MaxMetric


class LightningEvaluator(pl.LightningModule):
    def __init__(self, evaluator, lr=0.01, wd=5e-4):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["evaluator"])
        self.evaluator = evaluator
        self.train_acc = Accuracy(task="multiclass", num_classes=evaluator.out_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=evaluator.out_size)
        self.val_acc_best = MaxMetric()
        self.test_acc = Accuracy(task="multiclass", num_classes=evaluator.out_size)
        self.test_acc_per_cls = Accuracy(
            task="multiclass", num_classes=evaluator.out_size, average=None
        )

    def forward(self, g, features, edge_weight=None, g_type="cond"):
        return self.evaluator(g, features, edge_weight, g_type)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight, "cond")[data.ndata["train_mask"]]
        y = data.ndata["label"][data.ndata["train_mask"]]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight, "orig")[data.ndata["val_mask"]]
        y = data.ndata["label"][data.ndata["val_mask"]]
        val_acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.val_acc_best(val_acc)

        self.log(
            "val_acc",
            val_acc,
            batch_size=1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            batch_size=1,
            sync_dist=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.evaluator.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

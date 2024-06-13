import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MaxMetric


class Validator(pl.LightningModule):
    def __init__(self, backbone, lr=0.01, wd=5e-4):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.backbone = backbone
        self.train_acc = Accuracy(task="multiclass", num_classes=backbone.out_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=backbone.out_size)
        self.val_acc_best = MaxMetric()
        self.test_acc = Accuracy(task="multiclass", num_classes=backbone.out_size)
        self.test_acc_per_cls = Accuracy(
            task="multiclass", num_classes=backbone.out_size, average=None
        )

    def forward(self, g, features, edge_weight=None):
        return self.backbone(g, features, edge_weight)

    def encode(self, g, features, edge_weight=None):
        return self.backbone.encode(g, features, edge_weight)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight)[data.ndata["train_mask"]]
        y = data.ndata["label"][data.ndata["train_mask"]]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "validator/train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight)[data.ndata["val_mask"]]
        y = data.ndata["label"][data.ndata["val_mask"]]
        val_acc = self.val_acc(y_hat.softmax(dim=-1), y)
        self.val_acc_best(val_acc)

        self.log(
            "validator/val_acc",
            val_acc,
            batch_size=1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validator/acc_best",
            self.val_acc_best.compute(),
            batch_size=1,
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight)[data.ndata["test_mask"]]
        y = data.ndata["label"][data.ndata["test_mask"]]
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log("validator/test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return self.test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.backbone.parameters(), lr=self.lr, weight_decay=self.wd
        )

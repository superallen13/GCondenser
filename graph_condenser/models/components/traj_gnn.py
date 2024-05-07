from graph_condenser.models.backbones.lightning_gnn import LightningGNN

import torch
import torch.nn.functional as F


class TrajGNN(LightningGNN):
    def __init__(
        self,
        backbone: torch.nn.Module,
        lr: float = 0.001,
        wd: float = 5e-4,
        param_save_interval: int = 10,
    ):
        super().__init__(backbone, lr, wd)
        self.timestamps = []

    def on_train_start(self):
        # add the initial state of the model parameters
        self.timestamps.append([p.detach().cpu() for p in self.backbone.parameters()])

    def training_step(self, data, batch_idx):
        y_hat = self(data, data.ndata["feat"])[data.ndata["train_mask"]]
        y = data.ndata["label"][data.ndata["train_mask"]]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def on_train_epoch_end(self):
        # save the model parameters every param_save_interval epochs
        if (self.current_epoch + 1) % self.hparams.param_save_interval == 0:
            self.timestamps.append([p.detach().cpu() for p in self.backbone.parameters()])

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
        )
        return opt

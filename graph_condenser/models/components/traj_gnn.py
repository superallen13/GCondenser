from graph_condenser.models.backbones.lightning_gnn import LightningGNN

import math

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
        self.param_save_interval = param_save_interval
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
        if (self.current_epoch + 1) % self.param_save_interval == 0:
            self.timestamps.append(
                [p.detach().cpu() for p in self.backbone.parameters()]
            )

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        return opt


# neighborhood-based difficulty measurer
def neighborhood_difficulty_measurer(g):
    edges = g.edges()
    label = g.ndata["label"]
    neighbor_labels = label[edges[1]]
    edge_features = torch.stack([edges[0], neighbor_labels], dim=0)
    edge_features = edge_features.t()
    index, count = torch.unique(
        edge_features, sorted=True, return_counts=True, dim=0
    )
    neighbor_class = torch.sparse_coo_tensor(index.T, count) 
    neighbor_class = neighbor_class.to_dense().float()
    neighbor_class = neighbor_class[g.ndata["train_mask"]]
    neighbor_class = F.normalize(neighbor_class, p=1, dim=1)
    neighbor_entropy = (
        -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))
    )  # Avoid log(0)
    local_difficulty = neighbor_entropy.sum(1)
    return local_difficulty


def difficulty_measurer(g):
    local_difficulty = neighborhood_difficulty_measurer(g)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes(g):
    node_difficulty = difficulty_measurer(g)
    _, indices = torch.sort(node_difficulty)
    train_idx = g.ndata["train_mask"].nonzero().squeeze()
    sorted_trainset = train_idx[indices]
    return sorted_trainset


def training_scheduler(lam, t, T, scheduler="geom"):
    if scheduler == "linear":
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == "root":
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == "geom":
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


class TrajCL(LightningGNN):
    def __init__(
        self,
        backbone: torch.nn.Module,
        lr: float = 0.001,
        wd: float = 5e-4,
        param_save_interval: int = 10,
        lamb: float = 0.75,
        scheduler: str = "root",
        T: int = 1500,
    ):
        super().__init__(backbone, lr, wd)
        self.timestamps = []
        self.param_save_interval = param_save_interval
        self.lamb = lamb
        self.scheduler = scheduler
        self.T = T
        self.sorted_trainset = None

    def on_train_start(self):
        # add the initial state of the model parameters
        self.timestamps.append([p.detach().cpu() for p in self.backbone.parameters()])

    def training_step(self, data, batch_idx):
        if self.sorted_trainset is None:
            self.sorted_trainset = sort_training_nodes(data)
        sorted_trainset = self.sorted_trainset

        size = training_scheduler(self.lamb, self.current_epoch, self.T, self.scheduler)
        training_subset = sorted_trainset[:int(size * sorted_trainset.shape[0])]

        y_hat = self(data, data.ndata["feat"])[training_subset]
        y = data.ndata["label"][training_subset]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def on_train_epoch_end(self):
        # save the model parameters every param_save_interval epochs
        if (self.current_epoch + 1) % self.param_save_interval == 0:
            self.timestamps.append(
                [p.detach().cpu() for p in self.backbone.parameters()]
            )

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        return opt

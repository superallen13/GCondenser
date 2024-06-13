import dgl.sparse as dglsp
from dgl.sparse import SparseMatrix
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, MaxMetric


class LightningGNN(pl.LightningModule):
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

        # handle soft lables for the condensed graph
        soft_label = data.ndata["soft_label"] if "soft_label" in data.ndata else None
        if soft_label is not None:
            y = soft_label[data.ndata["train_mask"]]
            loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(
                F.log_softmax(y_hat, dim=1), y
            )
        else:
            y = data.ndata["label"][data.ndata["train_mask"]]
            loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight)[data.ndata["val_mask"]]
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

    def test_step(self, data, batch_idx):
        edge_weight = data.edata["weight"] if "weight" in data.edata else None
        y_hat = self(data, data.ndata["feat"], edge_weight)[data.ndata["test_mask"]]
        y = data.ndata["label"][data.ndata["test_mask"]]
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return self.test_acc

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.backbone.parameters(), lr=self.lr, weight_decay=self.wd
        )


class LightningGNNSparse(LightningGNN):
    def __init__(self, backbone, lr=0.01, wd=5e-4):
        super().__init__(backbone, lr, wd)
        self.train_adj = None
        self.val_adj = None
        self.test_adj = None

        self.train_adj = None
        self.val_adj = None
        self.test_adj = None

    def forward(self, A: SparseMatrix, X):
        return self.backbone(A, X)

    def training_step(self, data, batch_idx):
        if self.train_adj is None:
            dense_adj = data.adj
            adj_indices = dense_adj.nonzero(as_tuple=False).t()
            adj_values = dense_adj[adj_indices[0], adj_indices[1]]
            adj = dglsp.spmatrix(
                adj_indices.to(self.device),
                adj_values.to(self.device),
                (data.num_nodes(), data.num_nodes()),
            )
            self.train_adj = adj
        adj = self.train_adj

        y_hat = self(adj, data.ndata["feat"])[data.ndata["train_mask"]]
        y = data.ndata["label"][data.ndata["train_mask"]]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, data, batch_idx):
        if self.val_adj is None:
            edges = data.edges()
            adj_indices = torch.vstack(edges)
            adj_values = torch.ones(adj_indices.shape[1])
            A = dglsp.spmatrix(
                adj_indices.to(self.device),
                adj_values.to(self.device),
                shape=(data.num_nodes(), data.num_nodes()),
            )
            degs = dglsp.sum(A, dim=1).to_dense().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm_diag = torch.diag(norm)
            A_norm_dense = norm_diag @ A.to_dense() @ norm_diag
            A_norm_indices = A_norm_dense.nonzero(as_tuple=False).t()
            A_norm_values = A_norm_dense[A_norm_indices[0], A_norm_indices[1]]
            A_norm = dglsp.spmatrix(
                A_norm_indices.to(self.device),
                A_norm_values.to(self.device),
                shape=(data.num_nodes(), data.num_nodes()),
            )
            self.val_adj = A_norm
        adj = self.val_adj
        y_hat = self(adj, data.ndata["feat"])[data.ndata["val_mask"]]
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

    def test_step(self, data, batch_idx):
        if self.test_adj is None:
            edges = data.edges()
            adj_indices = torch.vstack(edges)
            adj_values = torch.ones(adj_indices.shape[1])
            A = dglsp.spmatrix(
                adj_indices.to(self.device),
                adj_values.to(self.device),
                shape=(data.num_nodes(), data.num_nodes()),
            )
            degs = dglsp.sum(A, dim=1).to_dense().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            norm_diag = torch.diag(norm)
            A_norm_dense = norm_diag @ A.to_dense() @ norm_diag
            A_norm_indices = A_norm_dense.nonzero(as_tuple=False).t()
            A_norm_values = A_norm_dense[A_norm_indices[0], A_norm_indices[1]]
            A_norm = dglsp.spmatrix(
                A_norm_indices.to(self.device),
                A_norm_values.to(self.device),
                shape=(data.num_nodes(), data.num_nodes()),
            )
            self.test_adj = A_norm
        adj = self.test_adj
        y_hat = self(adj, data.ndata["feat"])[data.ndata["test_mask"]]
        y = data.ndata["label"][data.ndata["test_mask"]]
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log("test_acc", self.test_acc, prog_bar=True, on_step=False, on_epoch=True)
        return self.test_acc

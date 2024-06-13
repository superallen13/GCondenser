import torch
import torch.nn as nn
import dgl.nn.pytorch.conv as dglnn
import torch.nn.functional as F
import dgl.sparse as dglsp
from torch.nn.parameter import Parameter
from dgl.sparse import SparseMatrix


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.bias = None
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data.T)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, A: SparseMatrix, X):
        support = X @ self.weight
        output = dglsp.matmul(A, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNSparse(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.out_size = out_size
        self.layers = nn.ModuleList([])
        if nlayers == 1:
            self.layers.append(GraphConvolution(in_size, out_size))
        else:
            self.layers.append(GraphConvolution(in_size, hid_size))
            for _ in range(nlayers - 2):
                self.layers.append(GraphConvolution(hid_size, hid_size))
            self.layers.append(GraphConvolution(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameter()

    def reset_parameter(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, A: SparseMatrix, X):
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(A, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class GCN(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.out_size = out_size
        self.layers = nn.ModuleList([])
        if nlayers == 1:
            self.layers.append(dglnn.SGConv(in_size, out_size))
        else:
            self.layers.append(dglnn.SGConv(in_size, hid_size))
            for _ in range(nlayers - 2):
                self.layers.append(dglnn.SGConv(hid_size, hid_size))
            self.layers.append(dglnn.SGConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def encode(self, g, features, edge_weight=None):
        h = features
        for layer in list(self.layers)[:-1]:
            h = layer(g, h, edge_weight=edge_weight)
            h = F.relu(h)
        return h


def main():
    # python graph_condenser/models/backbones/gcn.py --dataset citeseer --observe_mode transductive --repeat 5
    import argparse
    import logging
    import os
    import time

    import rootutils
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from graph_condenser.data.utils import prepare_graph
    from graph_condenser.data.datamodule import DataModule, SAINTDataModule
    from graph_condenser.models.backbones.lightning_gnn import LightningGNN

    import torch
    from torchmetrics import Accuracy
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    logging.getLogger("lightning_utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    results = []
    graph = prepare_graph(args.dataset, args.data_dir)
    # datamodule = DataModule(graph, observe_mode=args.observe_mode)
    datamodule = SAINTDataModule(graph, observe_mode=args.observe_mode)

    for _ in range(args.repeat):
        gcn = GCN(graph.ndata["feat"].shape[1], graph.num_classes)
        model = LightningGNN(gcn, lr=0.001, wd=5e-4)
        checkpoint_callback = ModelCheckpoint(mode="max", monitor="val_acc")
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=500,
            check_val_every_n_epoch=1,
            callbacks=checkpoint_callback,
            enable_model_summary=True,
            logger=False,
        )
        start = time.time()
        trainer.fit(model, datamodule)
        print(f"Training time: {time.time() - start:.1f}s")
        ckpt_path = checkpoint_callback.best_model_path
        start = time.time()

        # test_acc = trainer.test(
        #     model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
        # )
        # results.append(test_acc[0]["test_acc"])

        edge_weight = graph.edata["weight"] if "weight" in graph.edata else None
        gcn = GCN(graph.ndata["feat"].shape[1], graph.num_classes)
        model = LightningGNN.load_from_checkpoint(ckpt_path, backbone=gcn).to("cpu")
        y_hat = model(graph, graph.ndata["feat"], edge_weight)[graph.ndata["test_mask"]]
        y = graph.ndata["label"][graph.ndata["test_mask"]]
        acc = Accuracy(task="multiclass", num_classes=graph.num_classes)
        test_acc = acc(y_hat.softmax(dim=-1), y)
        results.append(test_acc.item())
        print(f"Test accuracy: {test_acc.item() * 100:.1f}")

        print(f"Test time: {time.time() - start:.1f}s")
        os.remove(ckpt_path)  # delete the checkpoint files

    if len(results) > 1:
        std_dev = torch.std(torch.tensor(results)).item() * 100
    else:
        std_dev = 0.0

    print(
        f"{args.repeat} runs, average accuracy: {sum(results) / len(results) * 100:.1f}; std: {std_dev:.1f}"
    )


if __name__ == "__main__":
    main()

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.models.backbones.gnn import GNN

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn


class GCN(GNN):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(in_size, out_size)
        if nlayers == 1:
            self.layers.append(pygnn.SGConv(in_size, out_size))
        else:
            self.layers.append(pygnn.SGConv(in_size, hid_size))
            for _ in range(nlayers - 2):
                self.layers.append(pygnn.SGConv(hid_size, hid_size))
            self.layers.append(pygnn.SGConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_weight=edge_weight)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def encode(self, x, edge_index, edge_weight=None):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
        return h


def main():
    # python graph_condenser/models/backbones/gcn.py --dataset flickr --observe_mode inductive --repeat 5
    import argparse
    import logging
    import os
    import time

    from graph_condenser.data.utils import get_dataset
    from graph_condenser.data.datamodule import DataModule
    from graph_condenser.models.backbones.lightning_gnn import LightningGNN

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    results = []
    dataset = get_dataset(args.dataset, args.data_dir)
    graph = dataset[0]
    datamodule = DataModule(graph, observe_mode=args.observe_mode)
    for _ in range(args.repeat):
        gcn = GCN(dataset.num_features, dataset.num_classes)
        model = LightningGNN(gcn, lr=0.001, wd=5e-4)
        checkpoint_callback = ModelCheckpoint(mode="max", monitor="val_acc")
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=500,
            check_val_every_n_epoch=1,
            callbacks=checkpoint_callback,
            enable_model_summary=False,
            logger=False,
        )
        start = time.time()
        trainer.fit(model, datamodule)
        print(f"Training time: {time.time() - start:.1f}s")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        start = time.time()
        test_acc = trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
        )
        print(f"Test time: {time.time() - start:.1f}s")
        results.append(test_acc[0]["test_acc"])
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

import torch.nn as nn
import dgl.nn.pytorch.conv as dglnn


class SGC(nn.Module):
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
        self.layer = dglnn.SGConv(in_size, out_size, k=nlayers)

    def forward(self, g, features, edge_weight=None):
        h = features
        h = self.layer(g, h, edge_weight=edge_weight)
        return h

    def encode(self, g, features, edge_weight=None):
        return self.layer(g, features, edge_weight=edge_weight)


def main():
    # python graph_condenser/models/backbones/sgc.py --dataset flickr --observe_mode inductive --repeat 5
    import argparse
    import logging
    import os
    import time

    import rootutils

    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    from graph_condenser.data.utils import prepare_graph
    from graph_condenser.data.datamodule import DataModule
    from graph_condenser.models.backbones.lightning_gnn import LightningGNN

    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    results = []
    for _ in range(args.repeat):
        graph = prepare_graph(args.dataset, args.data_dir)

        datamodule = DataModule(graph, observe_mode=args.observe_mode)
        gcn = SGC(graph.ndata["feat"].size(1), graph.num_classes)

        model = LightningGNN(gcn, lr=0.01, wd=5e-4)
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
        ckpt_path = checkpoint_callback.best_model_path
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
        f"5 runs, average accuracy: {sum(results) / len(results) * 100:.1f}; std: {std_dev:.1f}"
    )


if __name__ == "__main__":
    main()

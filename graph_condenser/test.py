import os
import tqdm
import argparse
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import get_dataset
from graph_condenser.models.backbones.mlp import MLP
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.backbones.lightning_gnn import LightningGNN
from graph_condenser.models.backbones.lightning_model import LightningModel

from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.sgc import SGC
from graph_condenser.models.backbones.gat import GAT
from graph_condenser.models.backbones.sage import SAGE
from graph_condenser.models.backbones.appnp import APPNP
from graph_condenser.models.backbones.cheb import ChebNet

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

def get_sparse_graph(g, threshold=0.5):
    edge_weight = g.edata["weight"]
    edge_weight[edge_weight < threshold] = 0
    g.remove_edges((edge_weight == 0).nonzero(as_tuple=True)[0])
    del g.edata["weight"]
    return g

def main(models, g_cond, dataset, observe_mode, repeat=5):
    results = []
    for model_name in models:
        if model_name in ["GAT", "ChebNet", "SAGE"]:
            g_cond = get_sparse_graph(g_cond)
        datamodule = CondensedGraphDataModule(g_cond, dataset[0], observe_mode)
        for i in range(repeat):
            if model_name == "MLP":
                test_model = MLP(g_cond.ndata["feat"].shape[1], dataset.num_classes)
                model = LightningModel(test_model, 0.01, 5e-4)
            else:
                test_model = eval(model_name)(g_cond.ndata["feat"].shape[1], dataset.num_classes)
                model = LightningGNN(test_model, 0.01, 5e-4)
            checkpoint = ModelCheckpoint(
                monitor="val_acc",
                save_top_k=1,
                mode="max",
            )
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=600,
                log_every_n_steps=1,
                callbacks=[checkpoint],
                # enable_model_summary=False,
                # enable_progress_bar=False,
                logger=False,
            )
            trainer.fit(model, datamodule)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            test_acc = trainer.test(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
            )
            results.append(test_acc[0]["test_acc"])
            os.remove(ckpt_path)
        acc_mean = sum(results) / len(results)
        acc_std = torch.std(torch.tensor(results)).item()
        print(f"{model_name} test accuracy: {acc_mean * 100:.1f} ± {acc_std * 100:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--graph_path",
        type=str,
        default="",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    ckpt = torch.load(args.graph_path)
    g_cond = ckpt["g_cond"]

    # check if the self.feat_cond is the same with the feature of the condensed graph
    assert torch.allclose(g_cond.ndata["feat"], ckpt["state_dict"]["feat_cond"])

    models = ["SAGE"]

    dataset = get_dataset(args.dataset, args.data_dir)

    main(models, g_cond, dataset, args.observe_mode, args.repeat)

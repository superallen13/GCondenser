import os
import tqdm
import argparse
import rootutils
import logging

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

import pandas as pd
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


def main(exp_name, models, g_cond, dataset, observe_mode, repeat=5):
    # make the results as a dataframe
    df = pd.DataFrame(columns=models)

    for model_name in models:
        results = []
        if model_name in ["GAT", "ChebNet", "SAGE"] and "weight" in g_cond.edata:
            g_cond = get_sparse_graph(g_cond)
        datamodule = CondensedGraphDataModule(g_cond, dataset[0], observe_mode)
        for i in range(repeat):
            if model_name == "MLP":
                test_model = MLP(g_cond.ndata["feat"].shape[1], dataset.num_classes)
                model = LightningModel(test_model, 0.01, 5e-4)
            else:
                test_model = eval(model_name)(
                    g_cond.ndata["feat"].shape[1], dataset.num_classes
                )
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
                enable_model_summary=False,
                enable_progress_bar=True,
                logger=False,
            )
            trainer.fit(model, datamodule)
            ckpt_path = trainer.checkpoint_callback.best_model_path
            test_acc = trainer.test(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
            )
            results.append(test_acc[0]["test_acc"])
            os.remove(ckpt_path)
        acc_mean = sum(results) / len(results)
        acc_std = torch.std(torch.tensor(results)).item()
        print(f"{model_name} test accuracy: {acc_mean * 100:.1f} ± {acc_std * 100:.1f}")
        df.loc[exp_name, model_name] = acc_mean * 100
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--graph_dir", type=int, default="./graphs")
    parser.add_argument("--device", type=int, default="cuda:0")
    parser.add_argument("--data_dir", type=int, default="./data")
    args = parser.parse_args()

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    pl.seed_everything(args.seed, workers=True)

    models = ["MLP", "SGC", "GCN", "GAT", "ChebNet", "SAGE", "APPNP"]
    dfs = []

    # load each graph file in the directory
    for graph_file in tqdm.tqdm(os.listdir(args.graph_dir)):
        if graph_file.endswith(".ckpt"):
            print(f"Processing {graph_file}")
            ckpt = torch.load(os.path.join(args.graph_dir, graph_file))
            g_cond = ckpt["g_cond"]
            exp_name = graph_file.split(".")[0]
            dataset = exp_name.split("-")[0]
            observe_mode = exp_name.split("-")[1]
            budget = exp_name.split("-")[2]
            backbone = exp_name.split("-")[3]
            condenser = exp_name.split("-")[4]
            dataset = get_dataset(dataset, args.data_dir)
            dfs.append(main(exp_name, models, g_cond, dataset, observe_mode, args.repeat))

    # merge the results from top to bottom
    df = pd.concat(dfs)
    df.to_csv("cross_test.csv")

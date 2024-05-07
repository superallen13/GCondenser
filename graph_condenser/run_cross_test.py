import os
import rootutils
import logging
import argparse

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

import tqdm
import wandb
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

WANDB_ENTITY = "ruihong-yilun"
PRINT_MAP = {
    "gcond": "GCond",
    "gcondx": "GCondX",
    "doscond": "DosCond",
    "doscondx": "DosCondX",
    "sgdd": "SGDD",
    "gcdm": "GCDM",
    "dm": "DM",
    "sfgc": "SFGC",
    "gcn": "GCN",
    "sgc": "SGC",
    "citeseer": "CiteSeer",
    "cora": "Cora",
    "pubmed": "PubMed",
    "arxiv": "Arxiv",
    "flickr": "Flickr",
    "reddit": "Reddit",
}
CONDENSERS = [
    "GCond",
    "GCondX",
    "DosCond",
    "DosCondX",
    "SGDD",
    "GCDM",
    "DM",
    "SFGC",
]
BACKBONES = ["SGC", "GCN"]
BUDGET_INDEX = {
    "citeseer": [30, 60, 120],
    "cora": [35, 70, 140],
    "pubmed": [15, 30, 60],
    "arxiv": [90, 454, 909],
    "flickr": [44, 223, 446],
    "reddit": [153, 769, 1539],
}


def get_sparse_graph(g, threshold=0.5):
    edge_weight = g.edata["weight"]
    edge_weight[edge_weight < threshold] = 0
    g.remove_edges((edge_weight == 0).nonzero(as_tuple=True)[0])
    del g.edata["weight"]
    return g


def cross_test(dataset, exp_info, test_models, repeat=5):
    # make the results as a dataframe

    test_restults = []
    observe_mode = exp_info["observation"]
    g_cond = torch.load(exp_info["ckpt"])["g_cond"]
    dataset = get_dataset(dataset, "./data")

    for model_name in test_models:
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
                enable_progress_bar=False,
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
        test_restults.append(acc_mean * 100)
    return test_restults


def main(args):
    pl.seed_everything(args.seed, workers=True)

    dataset = args.dataset
    budget = BUDGET_INDEX[dataset][args.budget - 1]

    print(f"Fetching data from Weights & Biases...")
    print(f"Dataset: {dataset}, Budget: {budget}")

    api = wandb.Api()
    project = "GCondenser-sweep"
    runs = api.runs(
        WANDB_ENTITY + "/" + project,
        filters={"group": {"$regex": f"{dataset}.*{budget}-.*"}},
    )

    val_acc_list = []
    test_acc_list = []
    condenser_list = []
    backbone_list = []
    observation_list = []
    ckpt_list = []

    for run in tqdm.tqdm(runs):
        try:
            val_acc_list.append(run.summary._json_dict["val/acc_best"])
        except:
            val_acc_list.append(None)

        try:
            test_acc_list.append(run.summary._json_dict["test/acc_mean"])
        except:
            test_acc_list.append(None)

        try:
            ckpt_list.append(run.summary._json_dict["trainer/ckpt_path"])
        except:
            ckpt_list.append(None)

        group = run.group
        observation_list.append(group.split("-")[1])
        backbone_list.append(group.split("-")[3])
        condenser_list.append(group.split("-")[4])

    runs_df = pd.DataFrame(
        {
            "val_acc": val_acc_list,
            "test_acc": test_acc_list,
            "condenser": condenser_list,
            "backbone": backbone_list,
            "observation": observation_list,
            "ckpt": ckpt_list,
        }
    )

    runs_df["val_acc"] *= 100
    runs_df["test_acc"] *= 100

    runs_df["condenser"] = runs_df["condenser"].map(PRINT_MAP)
    runs_df["backbone"] = runs_df["backbone"].map(PRINT_MAP)

    # only keep the best args.metric (val_acc or test_acc) for each condenser-backbone pair
    df = runs_df.groupby(["condenser", "backbone"]).apply(
        lambda x: x.loc[x[args.metric].idxmax()]
    )

    test_models = ["MLP", "SGC"]  # , "GCN", "GAT", "ChebNet", "SAGE", "APPNP"]

    cross_test_df = pd.DataFrame(
        columns=["Method", "Backbone"] + test_models + ["Avg."]
    )
    for condenser in CONDENSERS:
        for backbone in BACKBONES:
            # get the first row of the dataframe that matches the condenser and backbone
            exp_info = df[
                (df["condenser"] == condenser) & (df["backbone"] == backbone)
            ].iloc[0]
            cross_test_results = cross_test(dataset, exp_info, test_models, args.repeat)
            cross_test_results.append(sum(cross_test_results) / len(cross_test_results))
            # cross_test_results is a dict, the key is the model name, the value is the test accuracy
            cross_test_df.loc[len(cross_test_df)] = [
                condenser,
                backbone,
            ] + cross_test_results

    # covert the results to a dataframe
    cross_test_df.to_csv(f"./tabels/{dataset}_{budget}_cross_test.csv", index=False)

    # convert the results to a latex table, save to {dataset}-{budget}-cross-test.tex
    with open(f"{dataset}_{budget}_cross_test.tex", "w") as f:
        f.write(
            cross_test_df.to_latex(
                index=False,
                float_format="%.1f",
                column_format="l" * 2 + "c" * (len(test_models) + 1),
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--metric", type=str, default="val_acc")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    main(args)

import os
import wandb
import tqdm
import pandas as pd
import torch

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import prepare_graph
from benchmark.run_cross_test import cross_test

WANDB_ENTITY = "ruihong-yilun"
PRINT_MAP = {
    "gcond": "GCond",
    "gcondx": "GCondX",
    "doscond": "DosCond",
    "doscondx": "DosCondX",
    "sgdd": "SGDD",
    "gcdm": "GCDM",
    "gcdmx": "GCDMX",
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
    "GCDMX",
    "DM",
    "SFGC",
]
BACKBONES = ["SGC", "GCN"]
BUDGETS = {
    "citeseer": [30, 60, 120],
    "cora": [35, 70, 140],
    "pubmed": [15, 30, 60],
    "arxiv": [90, 454, 909],
    "flickr": [44, 223, 446],
    "reddit": [153, 769, 1539],
    "products": [612, 1225, 2449],
}


def fetch_info_from_wandb(dataset_name, budget_index, project="GCondenser-sweep"):
    budget = BUDGETS[dataset_name][budget_index - 1]

    print(f"Fetching data from Weights & Biases...")
    print(f"Dataset: {dataset_name}, Budget: {budget}")

    api = wandb.Api()
    runs = api.runs(
        WANDB_ENTITY + "/" + project,
        filters={"group": {"$regex": f"{dataset_name}.*{budget}-.*"}},
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
    return runs_df


def fetch_best_model_from_wandb(dataset_name, budget_index, backbone, condenser):
    budget = BUDGETS[dataset_name][budget_index - 1]

    print(f"Fetching data from Weights & Biases...")
    print(
        f"Dataset: {dataset_name}, Budget: {budget}, Backbone: {backbone}, Condenser: {condenser}"
    )

    api = wandb.Api()
    project = "GCB-overall"
    runs = api.runs(
        WANDB_ENTITY + "/" + project,
        filters={
            "group": {"$regex": f"{dataset_name}.*{budget}-{backbone}-{condenser}"}
        },
    )

    # find the run with the highest val_acc, and download the model by artifact
    best_val_acc = 0
    best_run = None
    for run in tqdm.tqdm(runs):
        try:
            # val_acc = run.summary._json_dict["val/acc_best"]
            val_acc = run.summary._json_dict["test/acc_mean"]
        except:
            val_acc = None

        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_run = run
    
    if best_run is None:
        print(f"No model found for {dataset_name} {budget} {backbone} {condenser}")
        return
    else:
        artifacts = best_run.logged_artifacts()
        best_model = [artifact for artifact in artifacts if artifact.type == 'model'][0]

        path = best_model.download()
        ckpt_path = os.path.join(path, "model.ckpt")
        return ckpt_path

def main():
    ckpt_path = fetch_best_model_from_wandb("flickr", 1, "gcn", "dm")
    test_models = ["MLP"]
    g_orig = prepare_graph("flickr", "./data")
    g_cond = torch.load(ckpt_path, map_location=torch.device('cpu'))["g_cond"]
    cross_test_results = cross_test(g_orig, g_cond, "inductive", test_models, 5)


if __name__ == "__main__":
    main()

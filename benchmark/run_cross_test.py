import os
from dgl import data
from numpy import test
import rootutils
import logging
import argparse
import datetime
import random

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import prepare_graph
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.backbones.lightning_gnn import LightningGNN
from graph_condenser.models.backbones.lightning_model import LightningModel
from graph_condenser.models.backbones import load_backbone

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


def get_sparse_graph(g, threshold=0.5):
    edge_weight = g.edata["weight"]
    edge_weight[edge_weight < threshold] = 0
    g.remove_edges((edge_weight == 0).nonzero(as_tuple=True)[0])
    del g.edata["weight"]
    return g


def cross_test(dataset_name, g_orig, g_cond, observe_mode, test_models, repeat=5):
    test_restults = []
    for model_name in test_models:
        results = []
        if model_name in ["GAT", "ChebNet", "SAGE"] and "weight" in g_cond.edata:
            g_cond = get_sparse_graph(g_cond)
        datamodule = CondensedGraphDataModule(g_cond, g_orig, observe_mode)
        test_model = load_backbone(model_name.lower())(
            g_orig.ndata["feat"].shape[1], g_orig.num_classes
        )
        for i in range(repeat):
            if model_name == "MLP":
                model = LightningModel(test_model, 0.01, 5e-4)
            else:
                model = LightningGNN(test_model, 0.01, 5e-4)
            # use current time as the checkpoint
            current_time = datetime.datetime.now()
            time_string = current_time.strftime("%Y%m%d-%H%M%S%f")
            rand_num = random.randint(100, 999)
            file_name = f"{time_string}_{rand_num}"
            checkpoint = ModelCheckpoint(
                dirpath=f"./{dataset_name}_checkpoints",
                monitor="val_acc",
                save_top_k=1,
                mode="max",
                filename=file_name,
            )
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=600,
                log_every_n_steps=1,
                check_val_every_n_epoch=1,
                callbacks=[checkpoint],
                enable_model_summary=False,
                enable_progress_bar=False,
                logger=False,
            )
            # trainer = Trainer(
            #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
            #     devices=1,
            #     max_epochs=300,
            #     enable_checkpointing=False,
            #     # callbacks=checkpoint_callback,
            #     enable_progress_bar=True,
            #     enable_model_summary=False,
            #     logger=False,
            #     # we don't need to validate the model
            #     limit_val_batches=0,
            #     num_sanity_val_steps=0,
            # )
            trainer.fit(model, datamodule)
            ckpt_path = checkpoint.best_model_path
            test_acc = trainer.test(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
            )
            # test_acc = trainer.test(
            #     model=model, datamodule=datamodule, verbose=False
            # )
            results.append(test_acc[0]["test_acc"])
            os.remove(ckpt_path)
            print(f"Repeat {i}, {model_name}: {test_acc[0]['test_acc'] * 100:.1f}")
        acc_mean = sum(results) / len(results)
        acc_std = torch.std(torch.tensor(results)).item()
        print(f"{model_name} test accuracy: {acc_mean * 100:.1f} ± {acc_std * 100:.1f}")
        test_restults.append(acc_mean * 100)
    return test_restults


def main(args):
    from utils import fetch_info_from_wandb, CONDENSERS, BACKBONES, BUDGETS
    
    pl.seed_everything(args.seed, workers=True)

    dataset_name = args.dataset
    budget_index = args.budget
    # runs_df = fetch_info_from_wandb(dataset_name, budget_index)

    # # only keep the best args.metric (val_acc or test_acc) for each condenser-backbone pair
    # df = runs_df.groupby(["condenser", "backbone"]).apply(
    #     lambda x: x.loc[x[args.metric].idxmax()]
    # )
    df = pd.read_csv(f"./benchmark/results/{dataset_name}_{budget_index}_best_val_acc.csv")

    # test_models = ["MLP", "SGC", "GCN", "GAT", "ChebNet", "SAGE", "APPNP"]

    test_models = [args.test_model]
    cross_test_df = pd.DataFrame(
        columns=["Method", "Backbone"] + test_models + ["Avg."]
    )
    g_orig = prepare_graph(dataset_name, "./data")
    # for condenser in CONDENSERS:
    #     for backbone in BACKBONES:
        # get the first row of the dataframe that matches the condenser and backbone
    condenser = args.condenser
    backbone = args.backbone

    exp_info = df[
        (df["condenser"] == condenser) & (df["backbone"] == backbone)
    ].iloc[0]
    print(exp_info)
    g_cond = torch.load(exp_info["ckpt"])["g_cond"]
    cross_test_results = cross_test(dataset_name, g_orig, g_cond, exp_info["observation"], test_models, args.repeat)
    cross_test_results.append(sum(cross_test_results) / len(cross_test_results))
    # cross_test_results is a dict, the key is the model name, the value is the test accuracy
    cross_test_df.loc[len(cross_test_df)] = [
        condenser,
        backbone,
    ] + cross_test_results

    if args.save_table:
        # covert the results to a dataframe
        budget = BUDGETS[dataset_name][budget_index - 1]
        cross_test_df.to_csv(f"./tables/{dataset_name}_{budget}_{condenser}_{backbone}_{args.test_model}_cross_test.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--condenser", type=str, default="")
    parser.add_argument("--test_model", type=str, default="")
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--metric", type=str, default="val_acc")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--save_table", action="store_true")
    args = parser.parse_args()

    main(args)

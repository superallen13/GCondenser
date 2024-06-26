import os
import logging
import argparse

from dgl.transforms import remove_edges
from networkx import edges
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import prepare_graph, get_npc, _graph_sampling
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.backbones import load_backbone
from graph_condenser.models.backbones.lightning_gnn import LightningGNN
from benchmark.utils import BUDGETS

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    dataset_name = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = prepare_graph(dataset_name, args.data_dir)

    budget = BUDGETS[dataset_name][args.budget_index - 1]
    npc = get_npc(g, budget, "original")
    g = g.to(device)
    g_cond = _graph_sampling(g, args.init_method, npc, args.observation)

    datamodule = CondensedGraphDataModule(g_cond, g, args.observation)
    gnn = load_backbone(args.backbone)
    model = LightningGNN(gnn(g.ndata["feat"].size(1), g.num_classes), lr=0.01, wd=5e-4)
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
    trainer.fit(model, datamodule)
    ckpt_path = checkpoint_callback.best_model_path
    test_acc = trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
    )
    os.remove(ckpt_path)  # delete the checkpoint files
    return test_acc[0]["test_acc"]


def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("List of integers must be separated by commas")


if __name__ == "__main__":
    # python benchmark/run_sampling.py --seeds 1,2,3,4,5 --dataset products --observation transductive --budget_index 2 --init_method randomChoice --backbone gcn
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_int_list, required=True)
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observation", type=str, default="transductive")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--budget_index", type=int, default=1)
    parser.add_argument("--init_method", type=str, default="kCenter")
    parser.add_argument("--backbone", type=str, default="gcn")
    args = parser.parse_args()

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    results = []
    for seed in args.seeds:
        pl.seed_everything(seed)
        results.append(main(args))
    # mean and std
    print(f"Mean: {sum(results) / len(results) * 100:.1f}")
    print(f"Std: {torch.tensor(results).std() * 100:.1f}")

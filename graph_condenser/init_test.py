import os
import tqdm
import argparse
import rootutils
import logging

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.data.utils import get_dataset, get_npc, _graph_sampling
from graph_condenser.models.backbones.sgc import SGC
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.lightning_gnn import LightningGNN


import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    dataset = get_dataset(args.dataset, args.data_dir)
    npc = get_npc(dataset, args.budget, args.label_distribution)
    g_init = _graph_sampling(dataset, args.sampling_method, npc)

    resutls = []
    datamodule = CondensedGraphDataModule(g_init, dataset[0], args.observe_mode)
    for _ in range(args.repeat):
        model = LightningGNN(
            eval(args.model)(g_init.ndata["feat"].size(1), dataset.num_classes),
            lr=0.01,
            wd=5e-4,
        )
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
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
        )
        resutls.append(test_acc[0]["test_acc"])

    print(
        f"Dataset: {args.dataset}, Model: {args.model}, Budget: {args.budget}, Sampling: {args.sampling_method}"
    )
    print(f"Mean: {sum(resutls) / len(resutls)}")
    print(
        f"Std: {sum([(x - sum(resutls) / len(resutls))**2 for x in resutls]) / len(resutls)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--label_distribution", type=str, default="original")
    parser.add_argument("--sampling_method", type=str, default="kcenter")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--model", type=str, default="SGC")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    pl.seed_everything(args.seed)

    main(args)

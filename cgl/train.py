import logging
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data.utils import get_streaming_datasets
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.lightning_gnn import LightningGNN
from graph_condenser.data.datamodule import GraphCondenserDataModule, CondensedGraphDataModule
from graph_condenser.utils import (
    instantiate_callbacks,
    instantiate_loggers,
)

import dgl
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.profilers import SimpleProfiler
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="./configs", config_name="cgl.yaml")
def main(cfg):
    pl.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, streaming_datasets = get_streaming_datasets(
        cfg.dataset_name, cfg.dataset_dir, cfg.cls_per_graph, cfg.split_ratio
    )

    memory_bank = []
    for i, incoming_graph in enumerate(streaming_datasets):
        datamodule = GraphCondenserDataModule(incoming_graph[0], cfg.condenser.observe_mode)
        condenser = hydra.utils.instantiate(cfg.condenser, dataset=incoming_graph)
        callbacks = instantiate_callbacks(cfg.get("callbacks"))
        timer = Timer()
        callbacks.append(timer)
        logger = instantiate_loggers(cfg.get("logger"))

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=callbacks,
            logger=logger,
            inference_mode=False,
        )

        trainer.fit(
            model=condenser,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )

        ckpt_path = trainer.checkpoint_callback.best_model_path
        g_cond = torch.load(ckpt_path)["g_cond"]
        memory_bank.append(g_cond)

    metrics = []
    model = GCN(dataset[0].ndata["feat"].size(1), dataset.num_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for j in range(len(streaming_datasets)):
        # use the condense and train (CaT) framework to train the model
        # train the model with the memory bank
        memory_graph = dgl.batch(memory_bank[: j + 1]).to(device)

        for _ in range(200):
            out = model(memory_graph, memory_graph.ndata["feat"])[:, : cfg.cls_per_graph * (j + 1)]
            loss = criterion(out, memory_graph.ndata["label"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        performance, forgetting = [], []
        # calculate average performance and average forgetting.
        for k, test_graph in enumerate(streaming_datasets[: j + 1]):
            g = test_graph[0].to(device)
            logits = model(g, g.ndata["feat"])[:, : cfg.cls_per_graph * (j + 1)]
            predict = logits.argmax(dim=-1)[g.ndata["test_mask"]]
            acc = (predict == g.ndata["label"][g.ndata["test_mask"]]).float().mean()
            performance.append(acc)

            print(f"Task {j:2d}: {acc * 100:.2f}|", end="")

            if j > 0 and k < j:
                forgetting.append(acc - metrics[-1][k])
        metrics.append(performance)
        AP = sum(performance) / len(performance)
        AF = sum(forgetting) / len(forgetting) if forgetting else 0
        print(f"AP: {AP * 100:.2f}, AF: {AF * 100:.2f}")


if __name__ == "__main__":
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    # suppress a warning "SLUM auto-requeueing enabled. Setting signal handlers."
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    main()

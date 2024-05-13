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
    dataset, streaming_datasets = get_streaming_datasets(
        cfg.dataset_name, cfg.dataset_dir, cfg.cls_per_graph, cfg.split_ratio
    )

    gcn = GCN(dataset[0].ndata["feat"].size(1), dataset.num_classes)
    model = LightningGNN(gcn, lr=0.01, wd=5e-4)
    memory_bank = []

    metrics = []
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
        memory_bank.append(torch.load(ckpt_path)["g_cond"])

    metrics = []
    for i, incoming_graph in enumerate(streaming_datasets):
        # use the condense and train (CaT) framework to train the model
        # train the model with the memory bank
        memory_graph = dgl.batch(memory_bank)
        datamodule = CondensedGraphDataModule(memory_graph, incoming_graph[0])

        checkpoint_callback = ModelCheckpoint(mode="max", monitor="val_acc")
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=200,
            check_val_every_n_epoch=1,
            callbacks=checkpoint_callback,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, datamodule)

        # test the model
        ckpt_path = trainer.checkpoint_callback.best_model_path
        model = LightningGNN.load_from_checkpoint(ckpt_path, backbone=gcn)

        performance, forgetting = [], []
        # calculate average performance and average forgetting.
        for j, test_graph in enumerate(streaming_datasets[: i + 1]):
            g = test_graph[0].to(model.device)
            logits = model(g, g.ndata["feat"])[:, : cfg.cls_per_graph * (i + 1) + 1]
            predict = logits.argmax(dim=-1)[g.ndata["test_mask"]]
            acc = (predict == g.ndata["label"][g.ndata["test_mask"]]).float().mean()
            performance.append(acc)

            print(f"Task {j}: {acc:.4f}  ", end="")

            if i > 0 and j < i:
                forgetting.append(acc - metrics[-1][j])
        metrics.append(performance)
        AP = sum(performance) / len(performance)
        AF = sum(forgetting) / len(forgetting) if forgetting else 0
        print(f"AP: {AP:.4f}, AF: {AF:.4f}")


if __name__ == "__main__":
    main()

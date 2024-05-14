import os
import logging
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from data.utils import get_streaming_datasets
from graph_condenser.data.datamodule import (
    GraphCondenserDataModule,
)
from graph_condenser.utils import (
    instantiate_callbacks,
    instantiate_loggers,
)

import wandb
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
import omegaconf


@hydra.main(version_base="1.3", config_path="./configs", config_name="cgl.yaml")
def main(cfg):
    pl.seed_everything(cfg.seed)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]

    if cfg.wandb:
        config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            project="GCB_CGL",
            group=f"${cfg.dataset_name}_${cfg.condenser.budget}_${cfg.condenser.init_method}_${hydra_cfg['runtime']['choices']['condenser']}",
            config=config,
        )

    streaming_data_path = os.path.join(
        cfg.dataset_dir, "streaming", f"{cfg.dataset_name}"
    )
    # if streaming_data_path does not exist, use the get method
    if not os.path.exists(streaming_data_path):
        streaming_datasets = get_streaming_datasets(
            cfg.dataset_name, cfg.dataset_dir, cfg.cls_per_graph, cfg.split_ratio
        )
        torch.save(streaming_datasets, streaming_data_path)
    else:
        streaming_datasets = torch.load(streaming_data_path)

    memory_bank = []
    for incoming_graph in streaming_datasets:
        datamodule = GraphCondenserDataModule(
            incoming_graph[0], cfg.condenser.observe_mode
        )
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

    # save the memory bank to the output dir
    torch.save(memory_bank, os.path.join(output_dir, "memory_bank.pt"))

    if cfg.wandb:
        wandb.log({"memory_bank_path": os.path.join(output_dir, "memory_bank.pt")})

if __name__ == "__main__":
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    # suppress a warning "SLUM auto-requeueing enabled. Setting signal handlers."
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

    main()

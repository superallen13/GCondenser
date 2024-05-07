import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import rootutils
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Callback, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.profilers import SimpleProfiler
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

torch.set_float32_matmul_precision("medium")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # FIXME...
    from graph_condenser.data.datamodule import GraphCondenserDataModule

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    profiler = SimpleProfiler(filename="profiler-report")

    # get the original graph dataset and initialise the condensed grpah
    log.info(f"Instantiating graph dataset <{cfg.dataset.dataset_name}>")
    dataset = hydra.utils.call(cfg.dataset)
    datamodule = GraphCondenserDataModule(dataset[0], cfg.condenser.observe_mode)

    log.info(f"Instantiating condenser <{cfg.condenser._target_}>")
    condenser: LightningModule = hydra.utils.instantiate(cfg.condenser, dataset=dataset)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    timer = Timer()
    callbacks.append(timer)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        logger=logger,
        inference_mode=False,
        profiler=profiler,
    )

    object_dict = {
        "cfg": cfg,
        "condenser": condenser,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=condenser,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
        )
        fit_profiler_report_path = os.path.join(profiler.dirpath, "fit-" + profiler.filename + ".txt")
        log.info("Fit profilter report path: " + fit_profiler_report_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        test_accs = trainer.test(
            model=condenser, datamodule=datamodule, ckpt_path=ckpt_path, verbose=False
        )[0]
        log.info(
            f"Test accuracy: {test_accs['test/acc_mean'] * 100:.1f} ± {test_accs['test/acc_std'] * 100:.1f}"
        )
        log.info(f"Best ckpt path: {ckpt_path}")
        test_profiler_report_path = os.path.join(profiler.dirpath, "test-" + profiler.filename + ".txt")
        log.info("Test profilter report path: " + test_profiler_report_path)

    log.info(f"Train time: {timer.time_elapsed('train'):.1f} s")
    log.info(f"Validation time: {timer.time_elapsed('validate'):.1f} s")
    log.info(f"Test time: {timer.time_elapsed('test'):.1f} s")

    # log ckpt path, train time, validation time, test time
    if logger:
        for logger in trainer.loggers:
            logger.log_metrics(
                {
                    "trainer/ckpt_path": ckpt_path,
                    "trainer/fit_profiler_report_path": fit_profiler_report_path,
                    "trainer/test_profiler_report_path": test_profiler_report_path,
                    "train/time": timer.time_elapsed("train"),
                    "val/time": timer.time_elapsed("validate"),
                    "test/time": timer.time_elapsed("test"),
                }
            )

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # suppress some logs from pytorch_lightning
    logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
    

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

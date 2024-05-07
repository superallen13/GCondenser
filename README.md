<p align="center">
  <img src="./figs/logo.png" />
</p>

<div align="center">
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)]() -->

</div>

`GCondenser` is a graph condensation (GC) toolkit, carefully designed for the graph condensation area 🚀. It can benchmark existing GC methods, accelerate the development of your own GC methods, and be directly used for downstream applications, such as graph continual learning, graph neural architecture search, and federated graph learning.

## Get Started
`GCondenser` mainly depends on the following packages:
```
dgl
torch
pytorch-lightning
ogb  # fetech ogb datasets
hydra  # configuration management
```

Some packages are optional if you would like to use some advanced features:
```
wandb  # wandb logger
hydra-optuna-sweeper  # hyperparameter search using optuna
hydra-submitit-launcher  # submit jobs via submitit in SLURM
```

## Usage of `GCondenser`
The main configuration file is `./config/train.yaml`, which includes settings for the dataset, condenser, trainer, and logger. `GCondenser` conducts various experiments using `Hydra`. The files located in the `./config/experiment/` folder are used to set dataset and condenser information. To quickly run an experiment, for example, use the following command:
```
python graph_condenser/train.py experiment=arxiv_gcond
```
If you would like to change the default hyperparameters, you can either directly modify the configuration file or pass them via the CLI. For example, to change the learning rate for updating the condensed graph's features to 0.01, run:
```
python graph_condenser/train.py experiment=arxiv_gcond condenser.opt_feat.lr=1e-2
```
For more information, please check the [Hydra documentation](https://hydra.cc/docs/intro/).

### Supported Datasets
For a list of supported datasets, please refer to our [supported datasets documentation](./docs/dataset.md). We are continuously adding more public datasets.

### Condenser
To effectively use `GCondenser`, you may need to lookup the following parameters of the Condenser class.

| Item           | Description                                                                                                   | Config Key                  |
| -------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------- |
| NPC            | `GCondenser` provides three node-per-class (NPC) initialisation methods: `original`, `balanced`, `stric`. | condenser.labe_distribution |
| initialisation | Node features of condensed graph can be initialised by `randomChoice` and `kCenter`                           | condenser.init_method       |
| train model    | The backbone model for condensing the original graph                                                          | condenser.gnn               |
| validate model | The model trained with condensed graph in the validation step                                                 | condenser.gnn_val           |
| test model     | The model trained with condensed graph in the test step                                                       | condenser.gnn_test |

## Add a New Graph Condenser
You can easily add new graph condensers by creating a new class that inherits from `graph_condenser.models.condenser.Condenser`. In this new class, you will need to implement a `training_step()` method to define how the condensed graph should be updated each epoch. Please check out our [step-by-step guide](./docs/new-condenser.md) for adding a new method.

## Hyperparamter Sweep by Optuna
Create a file in the `./configs/hparams_search/` directory. For example, there is a file named `adj_feat_optuna.yaml`. To run a hyperparameter sweep, execute the following command:
```
python graph_condenser/train.py experiment=arxiv_gcond hparams_search=adj_feat_optuna
```
For more information, please refer to the [Optuna Sweeper Plugin for Hydra](https://hydra.cc/docs/plugins/optuna_sweeper/).

## Reproducible results of the paper
To replicate the performance of the `GCond` method on the `ogbn-arxiv` dataset with the first budget using the `SGC` backbone model, run the following script with the appropriate flags:
```
bash scripts/experiment.sh -d arxiv -b 1 -m sgc -c gcond
```
This script initiates an Optuna sweep process to find the optimal learning rates for the adjacency matrix and features.

## Acknowledgement
We are deeply grateful to the following repositories, which have been immensely helpful in the development of this benchmark:
- [GCond and DosCond](https://github.com/ChandlerBang/GCond)
- [SGDD](https://github.com/RingBDStack/SGDD)
- [SFGC](https://github.com/Amanda-Zheng/SFGC)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

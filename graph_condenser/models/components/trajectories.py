import os
import random
import argparse
import logging

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import prepare_graph
from graph_condenser.data.datamodule import DataModule
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.sgc import SGC
from graph_condenser.models.components.traj_gnn import TrajGNN

import tqdm
import torch
from pytorch_lightning import Trainer


class TrajectoryBuffer:
    def __init__(
        self,
        model,
        dropout,
        lr,
        wd,
        epochs,
        param_save_interval,
        traj_save_interval,
        load_all,
        max_files,
        max_experts,
        traj_dir,
        device,
    ):
        self.param_save_interval = param_save_interval
        self.traj_save_interval = traj_save_interval
        self.load_all = load_all
        self.max_files = max_files
        self.max_experts = max_experts

        self.traj_dir = os.path.join(traj_dir, f"{model}_dropout_{dropout}_lr_{lr}_wd_{wd}_epochs_{epochs}_interval_{param_save_interval}")
        self.device = device

        self.buffer = []
        self.expert_files = []
        self.expert_idx = 0
        self.file_idx = 0
        self.expert_trajectory = None

    def load(self):
        expert_dir = self.traj_dir

        if self.load_all:
            n = 0
            while os.path.exists(
                os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
            ):
                self.buffer = self.buffer + torch.load(
                    os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
                )
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            random.shuffle(self.buffer)
        else:
            n = 0
            while os.path.exists(
                os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
            ):
                self.expert_files.append(
                    os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))
                )
                n += 1
            if n == 0:
                raise AssertionError("No buffers detected at {}".format(expert_dir))
            random.shuffle(self.expert_files)
            if self.max_files is not None:
                self.expert_files = self.expert_files[: self.max_files]
            self.buffer = torch.load(self.expert_files[self.file_idx])
            if self.max_experts is not None:
                self.buffer = self.buffer[: self.max_experts]
            random.shuffle(self.buffer)

    def next(self):
        if self.load_all:
            self.expert_trajectory = self.buffer[
                random.randint(0, len(self.buffer) - 1)
            ]
        else:
            self.expert_trajectory = self.buffer[self.expert_idx]
            self.expert_idx += 1
            if self.expert_idx == len(self.buffer):
                self.expert_idx = 0
                self.file_idx += 1
                if self.file_idx == len(self.expert_files):
                    self.file_idx = 0
                    random.shuffle(self.expert_files)
                # logging.info("loading file {}".format(self.expert_files[self.file_idx]))
                if self.max_files != 1:
                    del self.buffer
                    self.buffer = torch.load(self.expert_files[self.file_idx])
                if self.max_experts is not None:
                    self.buffer = self.buffer[: self.max_experts]
                random.shuffle(self.buffer)

    def get(self, epoch, rand_start=False):
        if rand_start:
            start = range(0, epoch + 1, self.param_save_interval)
            epoch = random.choice(start)
            epoch = epoch // self.param_save_interval
        else:
            epoch = epoch // self.param_save_interval
        return self.expert_trajectory[epoch]


def main(args):
    path = os.path.join(
        args.traj_dir,
        args.dataset,
        f"{args.backbone}_dropout_{args.dropout}_lr_{args.lr}_wd_{args.wd}_epochs_{args.epochs}_interval_{args.param_interval}",
    )

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    print(f"Saving trajectories to {path}")

    data = prepare_graph(args.dataset, args.data_dir)
    datamodule = DataModule(data, args.observe_mode)

    trajectories = []
    for it in tqdm.tqdm(range(args.num_experts)):
        if args.backbone == "gcn":
            gnn = GCN(
                data.ndata["feat"].shape[1],
                data.num_classes,
                args.hid_size,
                args.nlayers,
                args.dropout,
            )
        elif args.backbone == "sgc":
            gnn = SGC(
                data.ndata["feat"].shape[1],
                data.num_classes,
                args.hid_size,
                args.nlayers,
                args.dropout,
            )
        model = TrajGNN(
            gnn, lr=args.lr, wd=args.wd, param_save_interval=args.param_interval
        )
        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=args.epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            # we don't need to validate the model
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule)
        trajectories.append(model.timestamps)

        if len(trajectories) == args.file_size:
            n = 0
            while os.path.exists(os.path.join(path, f"replay_buffer_{n}.pt")):
                n += 1
            torch.save(
                trajectories,
                os.path.join(path, f"replay_buffer_{n}.pt"),
            )
            trajectories = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="citeseer")
    parser.add_argument("--observe_mode", type=str, default="transductive")
    parser.add_argument("--traj_dir", type=str, default="./data/traj")
    parser.add_argument("--backbone", type=str, choices=["gcn", "sgc"], default="gcn")
    parser.add_argument("--num_experts", type=int, default=200)
    parser.add_argument("--hid_size", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--param_interval", type=int, default=10)
    parser.add_argument("--file_size", type=int, default=10)

    args = parser.parse_args()

    # suppress some logs from pytorch_lightning
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    main(args)

import os
import rootutils
import argparse
import logging

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import get_g_cond, get_npc
from graph_condenser.models.backbones.gcn import GCN
from data.utils import get_streaming_datasets

import pandas as pd
import dgl
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    streaming_data_path = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}")
    os.makedirs(os.path.dirname(streaming_data_path), exist_ok=True)
    # if streaming_data_path does not exist, use the get method
    if not os.path.exists(streaming_data_path):
        # first check if the parents dir path exists
        streaming_datasets = get_streaming_datasets(
            args.dataset_name, args.data_dir, args.cls_per_graph, args.split_ratio
        )
        torch.save(streaming_datasets, streaming_data_path)
    else:
        streaming_datasets = torch.load(streaming_data_path)

    if args.condenser == "whole":
        memory_bank = [stream_dataset[0] for stream_dataset in streaming_datasets]
    elif args.condenser == "random":
        memory_bank = []
        for stream_dataset in streaming_datasets:
            g_orig = stream_dataset[0]
            g_orig.num_classes = stream_dataset.num_classes
            npc = get_npc(g_orig, 10, "original")
            replay_graph = get_g_cond(g_orig, "transductive", npc, args.condenser)
            memory_bank.append(replay_graph)
    elif args.condenser == "kcenter":
        memory_bank = []
        for stream_dataset in streaming_datasets:
            g_orig = stream_dataset[0]
            g_orig.num_classes = stream_dataset.num_classes
            npc = get_npc(g_orig, 10, "original")
            replay_graph = get_g_cond(g_orig, "transductive", npc, args.condenser)
            memory_bank.append(replay_graph)
    else:
        memory_bank = torch.load(args.memory_bank_path)

    gnn = GCN(
        streaming_datasets[-1][0].ndata["feat"].size(1),
        streaming_datasets[-1][0].ndata["label"].max().item() + 1,
        dropout=0.0,
    ).to(device)
    optimiser = torch.optim.Adam(gnn.parameters(), lr=0.005, weight_decay=5e-4)

    metrics = []
    APs = []
    for i, memory_graph in enumerate(memory_bank):
        memory_graph = dgl.batch(memory_bank[: i + 1]).to(device)

        best_val_acc = 0.0
        for epoch in range(300):
            edge_weight = (
                memory_graph.edata["weight"] if "weight" in memory_graph.edata else None
            )
            y_hat = gnn(memory_graph, memory_graph.ndata["feat"], edge_weight)[
                memory_graph.ndata["train_mask"]
            ]
            y = memory_graph.ndata["label"][memory_graph.ndata["train_mask"]]
            loss = F.cross_entropy(y_hat, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        performance, forgetting = [], []
        # calculate average performance and average forgetting.
        for j, test_graph in enumerate(streaming_datasets[: i + 1]):
            g = test_graph[0].to(device)
            y_hat = gnn(g, g.ndata["feat"])[g.ndata["test_mask"]][
                :, : args.cls_per_graph * (i + 1)
            ]
            y = g.ndata["label"][g.ndata["test_mask"]]
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
            performance.append(acc.item())
            print(f"T{j:2d}: {acc * 100:.1f}|", end="")
            if i > 0 and j < i:
                forgetting.append(acc - metrics[j][j])
        metrics.append(performance)
        AP = sum(performance) / len(performance)
        AF = sum(forgetting) / len(forgetting) if forgetting else 0
        APs.append(AP)
        print(f"AP: {AP * 100:.1f}, AF: {AF * 100:.1f}")
    return APs

    # performace_matrix_path = os.path.dirname(args.memory_bank_path)
    # torch.save(metrics, os.path.join(performace_matrix_path, "performance_matrix.pt"))
    # print(f"Save the performance matrix to the {performace_matrix_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="arxiv")
    parser.add_argument("--cls_per_graph", type=int, default=2)
    parser.add_argument("--split_ratio", type=float, default=0.6)
    parser.add_argument("--condenser", type=str, default="")
    parser.add_argument("--memory_bank_path", type=str, default="")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

    data = []  # Create a list to store data dictionaries
    for i in range(args.repeat):
        pl.seed_everything(i)
        APs = main(args)  # Call your function to get APs
        for i, AP in enumerate(APs):
            data.append({"task": i, "condenser": args.condenser, "AP": AP})

    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(data, columns=["task", "condenser", "AP"])

    print(df)
    result_path = f"./data/cgl_{args.dataset_name}_{args.condenser}.csv"
    df.to_csv(result_path, index=False)
    print(f"Save the APs to: {result_path}")

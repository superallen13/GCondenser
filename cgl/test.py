import os
import rootutils
import argparse

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.utils import get_g_cond, get_npc
from graph_condenser.models.backbones.gcn import GCN
from data.utils import get_streaming_datasets

import dgl
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


def main(args):
    pl.seed_everything(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "cgl/ckpt"

    streaming_data_path = os.path.join(args.data_dir, "streaming", args.dataset_name)
    streaming_datasets = torch.load(streaming_data_path)

    if args.mb_type == "cgm":
        memory_bank = torch.load(args.memory_bank_path)
    elif args.mb_type == "original":
        memory_bank = [stream_dataset[0] for stream_dataset in streaming_datasets]
    elif args.mb_type == "randomchoice":
        memory_bank = []
        for stream_dataset in streaming_datasets:
            npc = get_npc(stream_dataset, 10, "original")
            replay_graph = get_g_cond(stream_dataset, "transductive", npc, args.mb_type)
            memory_bank.append(replay_graph)
    elif args.mb_type == "kcenter":
        memory_bank = []
        for stream_dataset in streaming_datasets:
            npc = get_npc(stream_dataset, 10, "original")
            replay_graph = get_g_cond(stream_dataset, "transductive", npc, args.mb_type)
            memory_bank.append(replay_graph)
        
    gnn = GCN(
        streaming_datasets[-1][0].ndata["feat"].size(1),
        streaming_datasets[-1][0].ndata["label"].max().item() + 1,
        dropout=0.0,
    ).to(device)
    optimiser = torch.optim.Adam(gnn.parameters(), lr=0.005, weight_decay=5e-4)

    metrics = []
    for i, memory_graph in enumerate(memory_bank):
        memory_graph = dgl.batch(memory_bank[: i + 1]).to(device)

        best_val_acc = 0.0
        for epoch in range(200):
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

        #     # validate
        #     g = streaming_datasets[i][0].to(device)
        #     y_hat = gnn(g, g.ndata["feat"])[g.ndata["val_mask"]]
        #     y = g.ndata["label"][g.ndata["val_mask"]]
        #     val_acc = (y_hat.argmax(dim=-1) == y).float().mean()

        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         # save the check point file
        #         checkpoint = {
        #             "model_state_dict": gnn.state_dict(),
        #             "optimiser_state_dict": optimiser.state_dict(),
        #         }
        #         torch.save(checkpoint, checkpoint_path)
        #         # print(f'Checkpoint saved at epoch {epoch} with validation accuracy: {val_acc:.4f}')
        # # load the check point file
        # checkpoint = torch.load(checkpoint_path)
        # gnn.load_state_dict(checkpoint["model_state_dict"])
        # optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

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
        print(f"AP: {AP * 100:.1f}, AF: {AF * 100:.1f}")

    # performace_matrix_path = os.path.dirname(args.memory_bank_path)
    # torch.save(metrics, os.path.join(performace_matrix_path, "performance_matrix.pt"))
    # print(f"Save the performance matrix to the {performace_matrix_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--dataset_name", type=str, default="arxiv")
    parser.add_argument("--cls_per_graph", type=int, default=2)
    parser.add_argument("--mb_type", type=str, default="cgm")
    parser.add_argument(
        "--memory_bank_path",
        type=str,
        default="/scratch/itee/uqyliu71/GCondenser/multirun/2024-05-15/18-05-27/2/memory_bank.pt",
    )
    args = parser.parse_args()

    main(args)

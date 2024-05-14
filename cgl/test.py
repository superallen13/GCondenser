import os
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.models.backbones.gcn import GCN

import dgl
import torch
import pytorch_lightning as pl


def main(dataset_name, memory_bank_path, cls_per_graph=2):
    pl.seed_everything(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    streaming_data_path = os.path.join("data", "streaming", dataset_name)
    streaming_datasets = torch.load(streaming_data_path)

    memory_bank = torch.load(memory_bank_path)

    metrics = []
    model = GCN(
        streaming_datasets[0][0].ndata["feat"].size(1),
        streaming_datasets[0].num_classes,
    ).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4
    )
    criterion = torch.nn.CrossEntropyLoss()
    for j in range(len(streaming_datasets)):
        # use the condense and train (CaT) framework to train the model
        # train the model with the memory bank
        memory_graph = dgl.batch(memory_bank[: j + 1]).to(device)

        for _ in range(200):
            out = model(memory_graph, memory_graph.ndata["feat"])[
                :, : cls_per_graph * (j + 1)
            ]
            loss = criterion(out, memory_graph.ndata["label"])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        performance, forgetting = [], []
        # calculate average performance and average forgetting.
        for k, test_graph in enumerate(streaming_datasets[: j + 1]):
            g = test_graph[0].to(device)
            logits = model(g, g.ndata["feat"])[:, : cls_per_graph * (j + 1)]
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
    dataset_name = ""
    memory_bank_path = ""
    main(dataset_name, memory_bank_path)

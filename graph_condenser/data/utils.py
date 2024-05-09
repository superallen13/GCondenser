import random
from typing import List

from graph_condenser.data.datamodule import DataModule
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.lightning_gnn import LightningGNN

import torch
from torch_geometric.datasets import Planetoid, Reddit, Flickr
import torch_geometric.transforms as T
from pytorch_lightning import Trainer
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.preprocessing import StandardScaler


class StandardTransform(object):
    def __call__(self, data):
        feat = data.x
        feat_train = feat[data.train_mask]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        data.x = torch.tensor(scaler.transform(feat), dtype=feat.dtype)
        return data


def get_dataset(dataset_name, data_dir):
    dataset_name = dataset_name.lower()
    if dataset_name in ["citeseer", "cora", "pubmed"]:
        dataset = Planetoid(
            root=data_dir, name=dataset_name.lower(), transform=T.NormalizeFeatures()
        )
    elif dataset_name == "arxiv":
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=data_dir)
        split_idx = dataset.get_idx_split()
        dataset.data["train_index"] = split_idx["train"]
        dataset.data["val_index"] = split_idx["valid"]
        dataset.data["test_index"] = split_idx["test"]

        transform = T.Compose(
            [T.IndexToMask(replace=True), StandardTransform(), T.ToUndirected()]
        )
        dataset.data = transform(dataset.data)
    elif dataset_name == "flickr":
        dataset = Flickr(root=data_dir)

        transform = T.Compose(
            [T.NormalizeFeatures(), StandardTransform()]
        )
        dataset.data = transform(dataset.data)
    elif dataset_name == "reddit":
        dataset = Reddit(root=data_dir)

        transform = T.Compose(
            [T.NormalizeFeatures(), StandardTransform()]
        )
        dataset.data = transform(dataset.data)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset


def get_npc(dataset, budget, label_distribution):  # npc means nodes per class
    assert label_distribution in ["balanced", "original", "strict"]
    g = dataset[0]
    y_train = g.y[g.train_mask]
    classes = range(dataset.num_classes)
    if label_distribution == "balanced":
        cls_ratios = [1 / len(classes) for _ in classes]
    elif label_distribution == "original":
        cls_ratios = [(y_train == cls).float().mean().item() for cls in classes]
    elif label_distribution == "strict":
        cls_ratios = [(y_train == cls).float().mean().item() for cls in classes]
        expected_npc = [budget * ratio for ratio in cls_ratios]
        npc_rounded = [round(npc) for npc in expected_npc]
        gap = budget - sum(npc_rounded)
        ratio_diffs = [
            (i, abs(npc_rounded[i] - expected_npc[i])) for i in range(len(classes))
        ]
        ratio_diffs.sort(key=lambda x: x[1], reverse=True)
        for i in range(abs(gap)):
            class_index = ratio_diffs[i][0]
            npc_rounded[class_index] += 1 if gap > 0 else -1
        assert budget - sum(npc_rounded) == 0
        return npc_rounded
    # some class may have 0 training samples (e.g., ogbn-products)
    npc = [max(1, int(budget * ratio)) if ratio != 0 else 0 for ratio in cls_ratios]
    gap = budget - sum(npc)
    npc[npc.index(max(npc))] += gap
    return npc


def _graph_sampling(
    dataset,
    sampling_method,
    npc,
    observe_mode="transductive",
):
    method = sampling_method.lower()
    data = dataset[0]
    if method == "kcenter":
        datamodule = DataModule(data, observe_mode)
        backbone = GCN(dataset.num_features, dataset.num_classes)
        model = LightningGNN(backbone, lr=0.01, wd=5e-4)
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=50,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            # we don't need to validate the model
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule)

        data.to(model.device)
        emb = model.encode(data.x, data.edge_index)
        idx_selected = []
        for cls in range(dataset.num_classes):
            train_mask_cls = (data.y == cls) & data.train_mask
            idx = train_mask_cls.nonzero(as_tuple=True)[0]
            emb_cls = emb[train_mask_cls]
            mean_cls = torch.mean(emb_cls, dim=0, keepdim=True)
            dis = torch.cdist(emb_cls, mean_cls)[:, 0]
            _, idx_centers = torch.topk(dis, npc[cls], largest=False)
            idx_selected.append(idx[idx_centers])
        idx_selected = torch.cat(idx_selected)
        sampled_graph = data.subgraph(idx_selected)
    elif method == "randomchoice":
        idx_selected = []
        classes = data.y[data.train_mask].unique()
        for i, cls in enumerate(classes):
            train_mask_at_cls = (data.y == cls) & data.train_mask
            ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
            idx_selected += random.sample(ids_at_cls, k=npc[i])
        sampled_graph = data.subgraph(torch.tensor(idx_selected))
    else:
        raise ValueError("Sampling method is not found.")
    return sampled_graph


def get_g_cond(
    dataset,
    observe_mode: str,
    npc: List[int],
    init_method: str = "kCenter",
):
    g_cond = _graph_sampling(dataset, init_method, npc, observe_mode)

    g_cond.x = torch.nn.Parameter(g_cond.x, requires_grad=True)
    g_cond.train_mask = torch.ones(g_cond.num_nodes, dtype=torch.bool)

    # remove all edges and add self-loops
    g_cond.edge_index = torch.stack(
        [torch.arange(g_cond.num_nodes), torch.arange(g_cond.num_nodes)]
    )

    return g_cond

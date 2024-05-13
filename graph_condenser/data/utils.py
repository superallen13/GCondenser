import random
import logging
from typing import List
from collections import Counter

from graph_condenser.data.datamodule import DataModule, CondensedGraphDataModule
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.lightning_gnn import LightningGNN

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import dgl.transforms as T
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    CoraFullDataset,
    PubmedGraphDataset,
    FlickrDataset,
    RedditDataset,
    CoauthorCSDataset,
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
)
from dgl.data import AsNodePredDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler


class StandardTransform(object):
    def __call__(self, data):
        feat = data.ndata["feat"]
        feat_train = feat[data.ndata["train_mask"]]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        data.ndata["feat"] = torch.tensor(scaler.transform(feat), dtype=feat.dtype)
        return data


class MaskRetype(object):
    def __call__(self, data):
        data.ndata["train_mask"] = (
            data.ndata["train_mask"].clone().detach().to(dtype=torch.bool)
        )
        data.ndata["val_mask"] = (
            data.ndata["val_mask"].clone().detach().to(dtype=torch.bool)
        )
        data.ndata["test_mask"] = (
            data.ndata["test_mask"].clone().detach().to(dtype=torch.bool)
        )
        return data


def get_dataset(dataset_name, data_dir):
    dataset_name = dataset_name.lower()
    if dataset_name == "citeseer":
        dataset = CiteseerGraphDataset(
            raw_dir=data_dir,
            transform=T.Compose([T.AddSelfLoop()]),
        )
    elif dataset_name == "cora":
        dataset = CoraGraphDataset(
            raw_dir=data_dir,
            transform=T.Compose([T.AddSelfLoop()]),
        )
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset(
            raw_dir=data_dir,
            transform=T.Compose([T.AddSelfLoop()]),
        )
    elif dataset_name == "flickr":
        dataset = FlickrDataset(
            raw_dir=data_dir,
            transform=T.Compose(
                [
                    T.AddSelfLoop(),
                    MaskRetype(),
                    T.RowFeatNormalizer(),
                    StandardTransform(),
                ]
            ),
        )
    elif dataset_name == "reddit":
        dataset = RedditDataset(
            raw_dir=data_dir,
            transform=T.Compose(
                [T.AddSelfLoop(), T.RowFeatNormalizer(), StandardTransform()]
            ),
        )
    elif dataset_name == "arxiv":
        transform = T.Compose(
            [
                T.AddReverse(),
                T.AddSelfLoop(),
                MaskRetype(),
                T.RowFeatNormalizer(),
                StandardTransform(),
            ]
        )
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=data_dir))
        dataset.g = transform(dataset.g)
    elif dataset_name == "products":
        transform = T.Compose(
            [
                T.AddSelfLoop(),
                MaskRetype(),
                T.RowFeatNormalizer(),
                StandardTransform(),
            ]
        )
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-products", root=data_dir)
        )
        dataset.g = transform(dataset.g)
    elif dataset_name == "papers100m":
        transform = T.Compose([T.AddSelfLoop(), MaskRetype()])
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root=data_dir)
        )
        dataset.g = transform(dataset.g)
    elif dataset_name == "corafull":
        dataset = CoraFullDataset(raw_dir=data_dir, transform=T.AddSelfLoop())
    elif dataset_name == "a-computer":
        dataset = AmazonCoBuyComputerDataset(
            raw_dir=data_dir, transform=T.AddSelfLoop()
        )
    elif dataset_name == "a-photo":
        dataset = AmazonCoBuyPhotoDataset(raw_dir=data_dir, transform=T.AddSelfLoop())
    elif dataset_name == "c-cs":
        dataset = CoauthorCSDataset(raw_dir=data_dir, transform=T.AddSelfLoop())
    elif dataset_name == "c-physics":
        dataset = CoauthorPhysicsDataset(raw_dir=data_dir, transform=T.AddSelfLoop())
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    if dataset_name in ["corafull", "a-computer", "a-photo", "c-cs", "c-physics"]:
        # make train/val/test masks, 20 nodes per class for trianing, 500 for validation and the rest for testing
        data = dataset._graph
        y = data.ndata["label"]
        data.ndata["train_mask"] = torch.zeros(data.num_nodes(), dtype=torch.bool)
        data.ndata["val_mask"] = torch.zeros(data.num_nodes(), dtype=torch.bool)
        data.ndata["test_mask"] = torch.zeros(data.num_nodes(), dtype=torch.bool)
        for i in range(dataset.num_classes):
            idx = (y == i).nonzero(as_tuple=True)[0]
            idx = idx[torch.randperm(len(idx))]
            data.ndata["train_mask"][idx[:20]] = True
            data.ndata["val_mask"][idx[20:520]] = True
            data.ndata["test_mask"][idx[20:520]] = True
    return dataset


# def get_npc(dataset, budget, label_distribution):  # npc means nodes per class
#     assert label_distribution in ["balanced", "original", "strict"]
#     g = dataset[0]
#     y_train = g.ndata["label"][g.ndata["train_mask"]]
#     classes = range(dataset.num_classes)
#     if label_distribution == "balanced":
#         cls_ratios = [1 / len(classes) for _ in classes]
#     elif label_distribution == "original":
#         cls_ratios = [(y_train == cls).float().mean().item() for cls in classes]
#     elif label_distribution == "strict":
#         cls_ratios = [(y_train == cls).float().mean().item() for cls in classes]
#         expected_npc = [budget * ratio for ratio in cls_ratios]
#         npc_rounded = [round(npc) for npc in expected_npc]
#         gap = budget - sum(npc_rounded)
#         ratio_diffs = [
#             (i, abs(npc_rounded[i] - expected_npc[i])) for i in range(len(classes))
#         ]
#         ratio_diffs.sort(key=lambda x: x[1], reverse=True)
#         for i in range(abs(gap)):
#             class_index = ratio_diffs[i][0]
#             npc_rounded[class_index] += 1 if gap > 0 else -1
#         assert budget - sum(npc_rounded) == 0
#         return npc_rounded
#     # some class may have 0 training samples (e.g., ogbn-products)
#     npc = [max(1, int(budget * ratio)) if ratio != 0 else 0 for ratio in cls_ratios]
#     gap = budget - sum(npc)
#     npc[npc.index(max(npc))] += gap
#     return npc


def get_npc(dataset, budget, label_distribution):
    assert label_distribution in ["balanced", "original"]
    g = dataset[0]
    y_train = g.ndata["label"][g.ndata["train_mask"]]
    classes = np.unique(y_train)
    class_count = Counter(y_train.tolist())
    if label_distribution == "balanced":
        cls_ratios = {cls: 1 / len(classes) for cls in classes}
        npc = [int(budget * cls_ratios.get(c, 0)) for c in range(dataset.num_classes)]
        gap = budget - sum(npc)  # gap should between 0 and num_classes (inclusive)

        # select the most gap frequent classes and plus 1
        for c, _ in class_count.most_common(gap):
            npc[c] += 1
    elif label_distribution == "original":
        class_ratios = {cls: count / len(y_train) for cls, count in class_count.items()}
        # for each existing class in the classes, at least one node.
        npc = [
            max(1, int(budget * class_ratios[c])) if c in class_count else 0
            for c in range(dataset.num_classes)
        ]
        gap = budget - sum(npc)  # gap can be negative if budget is too small

        # match sure the label distribution is as close as possible to the original distribution
        if gap < 0:
            for _ in range(-gap):
                new_cls_ratios = {c: npc[c] / budget for c in classes}
                diff = {c: new_cls_ratios[c] - class_ratios[c] for c in classes}
                # find the c which is the class with the largest difference but npc[c] must be greater than 1
                for c in sorted(diff, key=diff.get, reverse=True):
                    if npc[c] > 1:
                        npc[c] -= 1
                        break
        elif gap > 0:
            for _ in range(gap):
                new_cls_ratios = {c: npc[c] / budget for c in classes}
                diff = {c: new_cls_ratios[c] - class_ratios[c] for c in classes}
                c = max(diff, key=diff.get)
                npc[c] += 1
    assert sum(npc) == budget
    return npc


def _graph_sampling(
    dataset,
    sampling_method,
    npc,
    observe_mode="transductive",
):
    g_orig = dataset[0]
    method = sampling_method.lower()
    if method == "kcenter":
        datamodule = DataModule(g_orig, observe_mode)
        backbone = GCN(g_orig.ndata["feat"].shape[1], dataset.num_classes)
        model = LightningGNN(backbone, lr=0.01, wd=5e-4)
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=100,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            # we don't need to validate the model
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule)

        g_orig.to(model.device)
        emb = model.encode(g_orig, g_orig.ndata["feat"])
        idx_selected = []
        for cls in range(dataset.num_classes):
            train_mask_cls = (g_orig.ndata["label"] == cls) & g_orig.ndata["train_mask"]
            idx = train_mask_cls.nonzero(as_tuple=True)[0]
            emb_cls = emb[train_mask_cls]
            mean_cls = torch.mean(emb_cls, dim=0, keepdim=True)
            dis = torch.cdist(emb_cls, mean_cls)[:, 0]
            _, idx_centers = torch.topk(dis, npc[cls], largest=False)
            idx_selected.append(idx[idx_centers])
        idx_selected = torch.cat(idx_selected)
        sampled_graph = dgl.node_subgraph(g_orig, idx_selected)
    elif method == "randomchoice":
        idx_selected = []
        classes = dataset.num_classes
        for i, cls in enumerate(range(classes)):
            train_mask_at_cls = (g_orig.ndata["label"] == cls) & g_orig.ndata[
                "train_mask"
            ]
            ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
            idx_selected += random.sample(ids_at_cls, k=npc[i])
        sampled_graph = dgl.node_subgraph(g_orig, idx_selected)
    elif method == "randomnoise":
        # TODO simplify this process, currently implementaion is not efficient
        idx_selected = []
        classes = g_orig.ndata["label"][g_orig.ndata["train_mask"]].unique()
        for i, cls in enumerate(classes):
            train_mask_at_cls = (g_orig.ndata["label"] == cls) & g_orig.ndata[
                "train_mask"
            ]
            ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
            idx_selected += random.sample(ids_at_cls, k=npc[i])
        sampled_graph = dgl.node_subgraph(g_orig, idx_selected)
        # change the sampled_graph's features to ranodm gussain noise
        sampled_graph.ndata["feat"] = torch.randn_like(sampled_graph.ndata["feat"])
    else:
        raise ValueError("Sampling method is not found.")
    return sampled_graph


def get_g_cond(
    dataset: dgl.data.DGLDataset,
    observe_mode: str,
    npc: List[int],
    init_method: str = "kCenter",
) -> dgl.DGLGraph:
    g_cond = _graph_sampling(dataset, init_method, npc, observe_mode)

    g_cond.ndata["feat"] = torch.nn.Parameter(g_cond.ndata["feat"], requires_grad=True)
    g_cond.ndata["train_mask"] = torch.ones(g_cond.num_nodes(), dtype=torch.bool)

    # remove all edges and add self-loops
    g_cond.remove_edges(list(range(g_cond.number_of_edges())))
    g_cond = dgl.add_self_loop(g_cond)

    return g_cond


def dataset_statistic(
    datasets=[
        "citeseer",
        "cora",
        "pubmed",
        "arxiv",
        "flickr",
        "reddit",
        "products",
        "corafull",
        "a-computer",
        "a-photo",
        "c-cs",
        "c-physics",
    ],
    data_dir="./data",
):
    """
    Return the statistics of the given datasets.
    Inoformation including the number of nodes, edges, features, classes,
    training nodes, validation nodes and test nodes.
    """
    for dataset_name in datasets:
        dataset = get_dataset(dataset_name, data_dir)
        data = dataset[0]
        num_nodes = data.num_nodes()
        num_edges = data.num_edges()
        num_features = data.ndata["feat"].shape[1]
        num_classes = dataset.num_classes
        num_train_nodes = data.ndata["train_mask"].sum().item()
        num_val_nodes = data.ndata["val_mask"].sum().item()
        num_test_nodes = data.ndata["test_mask"].sum().item()
        print(f"Dataset: {dataset_name}")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Number of features: {num_features}")
        print(f"Number of classes: {num_classes}")
        print(f"Number of training nodes: {num_train_nodes}")
        print(f"Number of validation nodes: {num_val_nodes}")
        print(f"Number of test nodes: {num_test_nodes}")
        print("---------------------------")


def main():
    # seed_everything(1024)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    dataset = get_dataset("cora", "./data")
    data = dataset[0]
    npc = get_npc(data, 35)
    x_cond, edges_cond, y_cond = get_g_cond(dataset, "transductive", 30, npc)

    g_cond = dgl.graph(edges_cond, num_nodes=x_cond.shape[0])
    g_cond.ndata["feat"] = x_cond
    g_cond.ndata["label"] = y_cond
    g_cond.ndata["train_mask"] = torch.ones(x_cond.shape[0], dtype=torch.bool)

    datamodule = CondensedGraphDataModule(data, g_cond, observe_mode="transductive")

    gnn = GCN(data.ndata["feat"].shape[1], dataset.num_classes)
    model = LightningGNN(gnn, lr=0.01, wd=5e-4)
    checkpoint_callback = ModelCheckpoint(mode="max", monitor="val_acc")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=500,
        check_val_every_n_epoch=1,
        callbacks=checkpoint_callback,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, datamodule)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    test_acc = trainer.test(
        model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
    )


if __name__ == "__main__":
    main()

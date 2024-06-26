import random
from collections import Counter

import rootutils
import numpy as np
import torch
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
from dgl.data.utils import add_nodepred_split
from pytorch_lightning import Trainer
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.data.datamodule import DataModule
from graph_condenser.models.backbones.gcn import GCN
from graph_condenser.models.backbones.lightning_gnn import LightningGNN


class StandardTransform(object):
    def __call__(self, data):
        feat = data.ndata["feat"]
        feat_train = feat[data.ndata["train_mask"]]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        data.ndata["feat"] = torch.tensor(scaler.transform(feat), dtype=feat.dtype)
        return data


class MaskRetype(object):
    """Retype the mask to bool type"""

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


def get_sparsified_k_hop_subgraph(g, seed_nodes, fanouts):
    """
    Modifed from the DGL tutorial:
    https://github.com/BarclayII/dgl/blob/new-tutorials/new-tutorial/L3_custom_sampler.ipynb
    """
    all_nodes = seed_nodes
    neighbor_nodes = seed_nodes
    for fanout in fanouts:
        src_nodes = seed_nodes.repeat_interleave(fanout)
        nodes, _ = dgl.sampling.random_walk(g, src_nodes, length=1)
        dst_nodes = nodes[:, 1]
        # When a random walk cannot continue because of lacking of successors (e.g. a
        # node is isolate, with no edges going out), dgl.sampling.random_walk will
        # pad the trace with -1.  Since OGB Products have isolated nodes, we should
        # look for the -1 entries and remove them.
        # mask = (dst_nodes != -1)
        # dst_nodes = dst_nodes[mask]
        
        # dst_nodes can not be repeated and should not appeared in all_nodes
        mask = ~torch.any(dst_nodes[:, None] == all_nodes, dim=-1)
        neighbor_nodes = dst_nodes[mask].unique()
        all_nodes = torch.cat([all_nodes, neighbor_nodes])
    sg = dgl.node_subgraph(g, all_nodes)
    return sg


def get_raw_dataset(dataset_name, dataset_dir):
    dataset_name = dataset_name.lower()
    if dataset_name == "citeseer":
        dataset = CiteseerGraphDataset(raw_dir=dataset_dir)
    elif dataset_name == "cora":
        dataset = CoraGraphDataset(raw_dir=dataset_dir)
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset(raw_dir=dataset_dir)
    elif dataset_name == "flickr":
        dataset = FlickrDataset(raw_dir=dataset_dir)
    elif dataset_name == "reddit":
        dataset = RedditDataset(raw_dir=dataset_dir)
    elif dataset_name == "arxiv":
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir), raw_dir=dataset_dir
        )
    elif dataset_name == "products":
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-products", root=dataset_dir), raw_dir=dataset_dir
        )
    elif dataset_name == "papers100m":
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-papers100M", root=dataset_dir), raw_dir=dataset_dir
        )
    elif dataset_name == "corafull":
        dataset = CoraFullDataset(raw_dir=dataset_dir)
    elif dataset_name == "a-computer":
        dataset = AmazonCoBuyComputerDataset(raw_dir=dataset_dir)
    elif dataset_name == "a-photo":
        dataset = AmazonCoBuyPhotoDataset(raw_dir=dataset_dir)
    elif dataset_name == "c-cs":
        dataset = CoauthorCSDataset(raw_dir=dataset_dir)
    elif dataset_name == "c-physics":
        dataset = CoauthorPhysicsDataset(raw_dir=dataset_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    if dataset_name in ["corafull", "a-computer", "a-photo", "c-cs", "c-physics"]:
        add_nodepred_split(dataset, [0.2, 0.4, 0.4])
    return dataset


def prepare_graph(dataset_name: str, dataset_dir: str):
    assert dataset_name in [
        "citeseer",
        "cora",
        "pubmed",
        "corafull",
        "a-computer",
        "a-photo",
        "c-cs",
        "flickr",
        "c-physics",
        "reddit",
        "arxiv",
        "products",
        "papers100m",
    ]  # only support these datasets for condensation
    raw_dataset = get_raw_dataset(dataset_name, dataset_dir)
    if dataset_name in ["reddit"]:
        transform = T.Compose([T.AddSelfLoop(), StandardTransform()])
    elif dataset_name in ["flickr"]:
        transform = T.Compose([MaskRetype(), T.AddSelfLoop(), StandardTransform()])
    elif dataset_name in ["arxiv"]:
        transform = T.Compose(
            [
                MaskRetype(),
                T.AddReverse(),
                T.AddSelfLoop(),
                StandardTransform(),
            ]
        )
    elif dataset_name in ["papers100m"]:
        raw_dataset[0].ndata["label"] = raw_dataset[0].ndata["label"].long()
        transform = T.Compose(
            [
                MaskRetype(),
                T.AddReverse(),
                T.AddSelfLoop(),
                StandardTransform(),
            ]
        )
    elif dataset_name in ["products"]:
        transform = T.Compose([MaskRetype(), T.AddSelfLoop()])
    else:
        # add self-loops
        transform = T.Compose([T.AddSelfLoop()])

    graph = raw_dataset[0]
    graph = transform(graph)
    setattr(graph, "num_classes", raw_dataset.num_classes)
    return graph


def get_npc(g, budget, label_distribution):
    y_train = g.ndata["label"][g.ndata["train_mask"]]
    classes = np.unique(y_train)
    class_count = Counter(y_train.tolist())
    if label_distribution == "balanced":
        cls_ratios = {cls: 1 / len(classes) for cls in classes}
        npc = [int(budget * cls_ratios.get(c, 0)) for c in range(g.num_classes)]
        gap = budget - sum(npc)  # gap should between 0 and num_classes (inclusive)

        # select the most gap frequent classes and plus 1
        for c, _ in class_count.most_common(gap):
            npc[c] += 1
    elif label_distribution == "original":
        class_ratios = {cls: count / len(y_train) for cls, count in class_count.items()}
        # for each existing class in the classes, at least one node.
        npc = [
            max(1, int(budget * class_ratios[c])) if c in class_count else 0
            for c in range(g.num_classes)
        ]
        gap = budget - sum(npc)  # gap can be negative if budget is too small

        # match sure the label distribution is as close as possible to the original distribution
        if gap < 0:
            for _ in range(-gap):
                new_cls_ratios = {c: npc[c] / budget for c in classes}
                diff = {c: new_cls_ratios[c] - class_ratios[c] for c in classes}
                sorted_classes = sorted(
                    diff, key=lambda x: float(diff[x]), reverse=True
                )
                # find the c which is the class with the largest difference but npc[c] must be greater than 1
                for c in sorted_classes:
                    if npc[c] > 1:
                        npc[c] -= 1
                        break
        elif gap > 0:
            candidate_labels = [c for c in classes if npc[c] < class_count[c]]
            for _ in range(gap):
                new_cls_ratios = {c: npc[c] / budget for c in candidate_labels}
                diff = {c: new_cls_ratios[c] - class_ratios[c] for c in candidate_labels}
                c = max(diff, key=lambda x: float(diff[x]))
                npc[c] += 1
                gap -= 1
                candidate_labels = [c for c in classes if npc[c] < class_count[c]]
    else:
        raise ValueError(f"Label distribution {label_distribution} not supported")
    assert sum(npc) == budget
    return npc


def _graph_sampling(
    g,
    method,
    npc,
    observe_mode="transductive",
):
    if method == "kcenter":
        datamodule = DataModule(g, observe_mode)
        backbone = GCN(g.ndata["feat"].shape[1], g.num_classes)
        model = LightningGNN(backbone, lr=0.01, wd=5e-4)
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=50,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            # we don't need to validate the model
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule)
        g = g.to(model.device)
        emb = model.encode(g, g.ndata["feat"])
        idx_selected = []
        for cls in range(g.num_classes):
            train_mask_cls = (g.ndata["label"] == cls) & g.ndata["train_mask"]
            idx = train_mask_cls.nonzero(as_tuple=True)[0]
            emb_cls = emb[train_mask_cls]
            mean_cls = torch.mean(emb_cls, dim=0, keepdim=True)
            dis = torch.cdist(emb_cls, mean_cls)[:, 0]
            _, idx_centers = torch.topk(dis, npc[cls], largest=False)
            idx_selected.append(idx[idx_centers])
        idx_selected = torch.cat(idx_selected)
        sampled_graph = dgl.node_subgraph(g, idx_selected)
    elif method == "random":
        idx_selected = []
        for i, cls in enumerate(range(g.num_classes)):
            train_mask_at_cls = (g.ndata["label"] == cls) & g.ndata["train_mask"]
            ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
            idx_selected += random.sample(ids_at_cls, k=npc[i])
        sampled_graph = dgl.node_subgraph(g, idx_selected)
    elif method == "noise":
        # TODO simplify this process, currently implementaion is not efficient
        idx_selected = []
        for i, cls in enumerate(range(g.num_classes)):
            train_mask_at_cls = (g.ndata["label"] == cls) & g.ndata["train_mask"]
            ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
            idx_selected += random.sample(ids_at_cls, k=npc[i])
        sampled_graph = dgl.node_subgraph(g, idx_selected)
        # change the sampled_graph's features to ranodm gussain noise
        sampled_graph.ndata["feat"] = torch.randn_like(sampled_graph.ndata["feat"])
    else:
        raise ValueError("Sampling method is not found.")
    return sampled_graph


def get_g_cond(
    g,
    observe_mode,
    npc,
    init_method="kcenter",
):
    g_cond = _graph_sampling(g, init_method, npc, observe_mode)

    g_cond.ndata["feat"] = torch.nn.Parameter(g_cond.ndata["feat"], requires_grad=True)
    g_cond.ndata["train_mask"] = torch.ones(g_cond.num_nodes(), dtype=torch.bool)

    # remove all edges and add self-loops
    g_cond.remove_edges(list(range(g_cond.number_of_edges())))
    g_cond = dgl.add_self_loop(g_cond)

    return g_cond


def dataset_statistic(dataset):
    """
    Return the statistics of the given datasets.
    Inoformation including the number of nodes, edges, features, classes,
    training nodes, validation nodes and test nodes.
    """
    data = dataset[0]
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()
    num_features = data.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    num_train_nodes = data.ndata["train_mask"].sum().item()
    num_val_nodes = data.ndata["val_mask"].sum().item()
    num_test_nodes = data.ndata["test_mask"].sum().item()
    print(f"Dataset: {dataset.name}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of training nodes: {num_train_nodes}")
    print(f"Number of validation nodes: {num_val_nodes}")
    print(f"Number of test nodes: {num_test_nodes}")
    print("---------------------------")


def main():
    dataset = get_raw_dataset("flickr", "./data")
    dataset_statistic(dataset)
    # npc = get_npc(data, 20, "original")
    # print(npc)
    # x_cond, edges_cond, y_cond = get_g_cond(dataset, "transductive", npc, "kCenter")


if __name__ == "__main__":
    main()

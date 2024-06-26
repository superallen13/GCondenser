import numpy as np
import torch
import dgl
import dgl.transforms as T
from dgl.data import DGLDataset
from dgl.data import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
    FlickrDataset,
    RedditDataset,
)
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.preprocessing import StandardScaler


class StreamingDataset(DGLDataset):
    def __init__(self, candidate):
        self.candidate = candidate
        super().__init__(name="streaming_dataset")
        self.num_classes = self.candidate.ndata["label"].max().item() + 1
        self.num_features = self.candidate.ndata["feat"].shape[1]

    def process(self):
        self.graph = self.candidate

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


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


def get_dataset(dataset_name, dataset_dir):
    dataset_name = dataset_name.lower()
    if dataset_name == "citeseer":
        dataset = CiteseerGraphDataset(
            raw_dir=dataset_dir,
        )
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
            DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir)
        )
    elif dataset_name == "products":
        dataset = AsNodePredDataset(
            DglNodePropPredDataset("ogbn-products", root=dataset_dir)
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return dataset


def get_streaming_datasets(dataset_name, dataset_dir, cls_per_graph=2, split_ratio=0.6):
    dataset = get_dataset(dataset_name, dataset_dir)
    original_graph = dataset[0]
    if dataset_name in ["arxiv"]:
        transform = T.Compose(
            [
                T.AddReverse(),
                MaskRetype(),
            ]
        )
        original_graph = transform(original_graph)
    elif dataset_name in ["flickr", "products"]:
        transform = T.Compose(
            [
                MaskRetype(),
            ]
        )
        original_graph = transform(original_graph)

    streaming_datasets = []

    labels = original_graph.ndata["label"].squeeze()
    num_tasks = dataset.num_classes // cls_per_graph

    # Create class lists for streaming
    streaming_cls_list = [
        list(range(i * cls_per_graph, (i + 1) * cls_per_graph)) for i in range(num_tasks)
    ]

    for streaming_cls in streaming_cls_list:
        print(f"Streaming classes: {streaming_cls}")
        # Select nodes that belong to the current class list
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for cls in streaming_cls:
            mask |= labels == cls
        streaming_graph = original_graph.subgraph(mask)

        # Split the streaming graph into train, validation, and test sets
        n = streaming_graph.number_of_nodes()
        perm = np.random.permutation(n)
        train_n = int(n * split_ratio)
        val_n = int((n - train_n) / 2)

        streaming_graph.ndata["train_mask"] = torch.zeros(n, dtype=torch.bool)
        streaming_graph.ndata["val_mask"] = torch.zeros(n, dtype=torch.bool)
        streaming_graph.ndata["test_mask"] = torch.zeros(n, dtype=torch.bool)

        streaming_graph.ndata["train_mask"][perm[:train_n]] = True
        streaming_graph.ndata["val_mask"][perm[train_n : train_n + val_n]] = True
        streaming_graph.ndata["test_mask"][perm[train_n + val_n :]] = True

        streaming_graph = dgl.add_self_loop(streaming_graph)
        streaming_dataset = StreamingDataset(streaming_graph)
        streaming_datasets.append(streaming_dataset)

    return streaming_datasets

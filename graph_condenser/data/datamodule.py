import dgl
from dgl.dataloading import SAINTSampler, DataLoader
import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(self, graph: dgl.DGLGraph, observe_mode="transductive"):
        super().__init__()
        self.g = graph
        assert observe_mode in ["transductive", "inductive"]
        self.transductive = True if observe_mode == "transductive" else False

    def train_dataloader(self):
        if self.transductive:
            return torch.utils.data.DataLoader([self.g], collate_fn=lambda xs: xs[0])
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["train_mask"])
            return torch.utils.data.DataLoader([seen_data], collate_fn=lambda xs: xs[0])

    def val_dataloader(self):
        if self.transductive:
            return torch.utils.data.DataLoader([self.g], collate_fn=lambda xs: xs[0])
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["val_mask"])
            return torch.utils.data.DataLoader([seen_data], collate_fn=lambda xs: xs[0])

    def test_dataloader(self):
        if self.transductive:
            return torch.utils.data.DataLoader([self.g], collate_fn=lambda xs: xs[0])
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["test_mask"])
            return torch.utils.data.DataLoader([seen_data], collate_fn=lambda xs: xs[0])


class SAINTDataModule(LightningDataModule):
    def __init__(self, graph: dgl.DGLGraph, observe_mode="transductive"):
        super().__init__()
        self.g = graph
        assert observe_mode in ["transductive", "inductive"]
        self.transductive = True if observe_mode == "transductive" else False
        self.sampler = SAINTSampler(mode="walk", budget=(800000, 2))
        self.num_iters = 1

    def train_dataloader(self):
        if self.transductive:
            return DataLoader(
                self.g,
                torch.arange(self.num_iters),
                self.sampler,
                num_workers=2,
            )
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["train_mask"])
            return DataLoader(
                seen_data,
                torch.arange(self.num_iters),
                self.sampler,
                num_workers=2,
            )

    def val_dataloader(self):
        if self.transductive:
            return DataLoader(
                self.g,
                torch.arange(self.num_iters),
                self.sampler,
                num_workers=2,
            )
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["val_mask"])
            return DataLoader(
                seen_data,
                torch.arange(self.num_iters),
                self.sampler,
                num_workers=2,
            )

    def test_dataloader(self):
        if self.transductive:
            return torch.utils.data.DataLoader([self.g], collate_fn=lambda xs: xs[0])
        else:
            seen_data = dgl.node_subgraph(self.g, self.g.ndata["test_mask"])
            return torch.utils.data.DataLoader([seen_data], collate_fn=lambda xs: xs[0])


class CondensedGraphDataModule(LightningDataModule):
    def __init__(self, g_cond, g_orig, observe_mode="transductive"):
        super().__init__()
        self.g_cond = g_cond
        self.g_orig = g_orig
        assert observe_mode in ["transductive", "inductive"]
        self.observe_mode = observe_mode

    def train_dataloader(self):
        return torch.utils.data.DataLoader([self.g_cond], collate_fn=lambda xs: xs[0])

    def val_dataloader(self):
        if self.observe_mode == "transductive":
            return torch.utils.data.DataLoader(
                [self.g_orig], collate_fn=lambda xs: xs[0]
            )
        else:
            data = dgl.node_subgraph(self.g_orig, self.g_orig.ndata["val_mask"])
            return torch.utils.data.DataLoader([data], collate_fn=lambda xs: xs[0])

    def test_dataloader(self):
        if self.observe_mode == "transductive":
            return torch.utils.data.DataLoader(
                [self.g_orig], collate_fn=lambda xs: xs[0]
            )
        else:
            data = dgl.node_subgraph(self.g_orig, self.g_orig.ndata["test_mask"])
            return torch.utils.data.DataLoader([data], collate_fn=lambda xs: xs[0])


class GraphCondenserDataModule(LightningDataModule):
    def __init__(self, g_orig, observe_mode="transductive"):
        super().__init__()
        self.g_orig = g_orig
        assert observe_mode in ["transductive", "inductive"]
        self.observe_mode = observe_mode

    def train_dataloader(self):
        if self.observe_mode == "transductive":
            seen_orig = self.g_orig
        else:
            seen_orig = dgl.node_subgraph(self.g_orig, self.g_orig.ndata["train_mask"])
        return torch.utils.data.DataLoader([seen_orig], collate_fn=lambda xs: xs[0])

    def val_dataloader(self):
        if self.observe_mode == "transductive":
            seen_orig = self.g_orig
        else:
            seen_orig = dgl.node_subgraph(self.g_orig, self.g_orig.ndata["val_mask"])
        return torch.utils.data.DataLoader([seen_orig], collate_fn=lambda xs: xs[0])

    def test_dataloader(self):
        return torch.utils.data.DataLoader([self.g_orig], collate_fn=lambda xs: xs[0])

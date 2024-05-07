import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.models.backbones.gnn import GNN

import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F


class SAGE(GNN):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(in_size, out_size)
        if nlayers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, aggregator_type="mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, aggregator_type="mean"))
            for _ in range(nlayers - 2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, aggregator_type="mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, aggregator_type="mean"))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

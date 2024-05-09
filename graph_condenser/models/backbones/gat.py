import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from graph_condenser.models.backbones.gnn import GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn


class GAT(GNN):
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
            self.layers.append(pygnn.GATConv(in_size, out_size, num_heads=1))
        else:
            self.layers.append(pygnn.GATConv(in_size, hid_size, num_heads=8))
            for _ in range(nlayers - 2):
                self.layers.append(pygnn.GATConv(hid_size * 8, hid_size, num_heads=8))
            self.layers.append(pygnn.GATConv(hid_size * 8, out_size, num_heads=1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight).flatten(1)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

from graph_condenser.models.backbones.gnn import GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn

class ChebNet(GNN):
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
            self.layers.append(pygnn.ChebConv(in_size, out_size, 2))
        else:
            self.layers.append(pygnn.ChebConv(in_size, hid_size, 2))
            for _ in range(nlayers - 2):
                self.layers.append(pygnn.ChebConv(hid_size, hid_size, 2))
            self.layers.append(pygnn.ChebConv(hid_size, out_size, 2))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
from graph_condenser.models.backbones.gnn import GNN

import torch
import torch.nn as nn
from torch.nn import Linear
import dgl.nn as dglnn
import torch.nn.functional as F

class APPNP(GNN):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(in_size, out_size)
        self.lin1 = Linear(in_size, hid_size)
        self.lin2 = Linear(hid_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.prop1 = dglnn.APPNPConv(k=10, alpha=0.1)
    
    def forward(self, g, features, edge_weight=None):
        h = features
        h = F.relu(self.lin1(h))
        h = self.dropout(h)
        h = self.lin2(h)
        h = self.prop1(g, h, edge_weight)
        return h
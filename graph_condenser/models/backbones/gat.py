import torch.nn as nn
import dgl.nn.pytorch.conv as dglnn
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.out_size = out_size
        self.layers = nn.ModuleList([])
        if nlayers == 1:
            self.layers.append(dglnn.GATConv(in_size, out_size, num_heads=1))
        else:
            self.layers.append(dglnn.GATConv(in_size, hid_size, num_heads=8))
            for _ in range(nlayers - 2):
                self.layers.append(dglnn.GATConv(hid_size * 8, hid_size, num_heads=8))
            self.layers.append(dglnn.GATConv(hid_size * 8, out_size, num_heads=1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight).flatten(1)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

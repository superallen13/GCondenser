import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch.conv as dglnn


class APPNP(nn.Module):
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
            self.layers.append(nn.Linear(in_size, out_size))
        else:
            self.layers.append(nn.Linear(in_size, hid_size))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
        self.prop1 = dglnn.APPNPConv(k=10, alpha=0.1)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        h = self.prop1(g, h, edge_weight)
        return h

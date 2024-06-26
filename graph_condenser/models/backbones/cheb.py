import torch.nn as nn
import dgl.nn.pytorch.conv as dglnn


class ChebNet(nn.Module):
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
            self.layers.append(dglnn.ChebConv(in_size, out_size, 2))
        else:
            self.layers.append(dglnn.ChebConv(in_size, hid_size, 2))
            for _ in range(nlayers - 2):
                self.layers.append(dglnn.ChebConv(hid_size, hid_size, 2))
            self.layers.append(dglnn.ChebConv(hid_size, out_size, 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features, edge_weight=None):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                # the activation function is relu in the dgl conv layer
                h = self.dropout(h)
        return h

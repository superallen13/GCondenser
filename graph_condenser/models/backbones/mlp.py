# write a mlp model
import torch.nn as nn
import torch.nn.functional as F
from graph_condenser.models.backbones.gnn import GNN

class MLP(GNN):
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
            self.layers.append(nn.Linear(in_size, out_size))
        else:
            self.layers.append(nn.Linear(in_size, hid_size))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(hid_size, hid_size))
            self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
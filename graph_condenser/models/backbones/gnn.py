import torch.nn as nn


class GNN(nn.Module):
    """
    Use stacked SGC layers plus activations as the GCN model.
    SGC layer do the AX first, and it is very curcial for the graph condensaiton.
    """
    def __init__(self, in_size: str, out_size: str):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layers = nn.ModuleList([])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    # def forward(self, g, features, edge_weight=None):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         h = layer(g, h, edge_weight=edge_weight)
    #         if i != len(self.layers) - 1:
    #             h = self.dropout(h)
    #     return h

    # def encode(self, g, features, edge_weight=None):
    #     # no dropout for the encoding process
    #     h = features
    #     for layer in self.layers[:-1]:
    #         h = layer(g, h, edge_weight=edge_weight)
    #     return h

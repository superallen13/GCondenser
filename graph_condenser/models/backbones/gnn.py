import torch.nn as nn


class GNN(nn.Module):
    def __init__(self, in_size: str, out_size: str):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.layers = nn.ModuleList([])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

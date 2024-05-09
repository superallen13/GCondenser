import torch.nn as nn
import torch_geometric.nn as pygnn


class SGCEvaluator(nn.Module):
    def __init__(
        self,
        in_size: str,
        out_size: str,
        hid_size: int = 256,
        nlayers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.nlayers = nlayers

        self.sgc_cond = pygnn.SGConv(in_size, out_size, K=nlayers, cached=True)
        self.sgc_orig = pygnn.SGConv(in_size, out_size, K=nlayers, cached=True)

    def reset_parameters(self):
        self.sgc_cond = pygnn.SGConv(
            self.in_size, self.out_size, K=self.nlayers, cached=True
        )

    def forward(self, x, edge_index, edge_weight=None, g_type="cond"):
        if g_type == "cond":
            return self.sgc_cond(x, edge_index, edge_weight=edge_weight)
        elif g_type == "orig":
            # share weights from the condensed model
            self.sgc_orig.lin = self.sgc_cond.lin
            return self.sgc_orig(x, edge_index, edge_weight=edge_weight)

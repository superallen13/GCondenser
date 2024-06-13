import torch.nn as nn
import dgl.nn as dglnn


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

        self.sgc_cond = dglnn.SGConv(in_size, out_size, k=nlayers, cached=True)
        self.sgc_orig = dglnn.SGConv(in_size, out_size, k=nlayers, cached=True)

    def reset_parameters(self):
        self.sgc_cond = dglnn.SGConv(
            self.in_size, self.out_size, k=self.nlayers, cached=True
        )

    def forward(self, g, features, edge_weight=None, g_type="cond"):
        if g_type == "cond":
            return self.sgc_cond(g, features, edge_weight=edge_weight)
        elif g_type == "orig":
            self.sgc_orig.fc = (
                self.sgc_cond.fc
            )  # share weights from the condensed model
            return self.sgc_orig(g, features, edge_weight=edge_weight)

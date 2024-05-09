import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureGenerator(nn.Module):

    def __init__(self, g, hid_size: int = 128, nlayers: int = 3):
        super(StructureGenerator, self).__init__()
        self.edges = g.edge_index
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(g.x.shape[1], hid_size))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hid_size))
        for i in range(nlayers - 2):
            self.layers.append(nn.Linear(hid_size, hid_size))
            self.bns.append(nn.BatchNorm1d(hid_size))
        self.layers.append(nn.Linear(hid_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)

    def forward(self, feat):
        src, dst = self.edges
        edge_emb = torch.add(feat[src], feat[dst]) / 2
        for ix, layer in enumerate(self.layers):
            edge_emb = layer(edge_emb)
            if ix != len(self.layers) - 1:
                edge_emb = self.bns[ix](edge_emb)
                edge_emb = F.relu(edge_emb)

        edge_weight = torch.sigmoid(edge_emb).flatten()
        edge_weight = edge_weight + 1e-6  # avoid zero weight

        self_loop_mask = src == dst
        edge_weight[self_loop_mask] = 1.0  # set self-loop weight to 1

        return edge_weight

    @torch.no_grad()
    def inference(self, feat):
        edge_weight = self.forward(feat)
        return edge_weight

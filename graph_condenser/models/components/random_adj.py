import torch.nn as nn
import torch.nn.functional as F
import torch


class RandomAdj(nn.Module):

    def __init__(self, nfeat, nhid=128, nlayers=3):
        super(RandomAdj, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat * 2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

    def forward(self, x):
        # a uniform distribtuion between 0 and 1
        adj = torch.rand(size=(x.size(0), x.size(0))).to("cuda:0")

        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    def inference(self, x):
        return self.forward(x)

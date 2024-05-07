from itertools import product

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def mx_inv(mx):
    U, D, V = torch.svd(mx)
    eps = 0.009
    D_min = torch.min(D)
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D > D_min] = 1 / D[D > D_min]
    else:
        D_1 = 1 / D
    # D_1 = 1 / D #.clamp(min=0.005)

    return U @ D_1.diag() @ V.t()


def mx_inv_sqrt(mx):
    # singular values need to be distinct for backprop
    U, D, V = torch.svd(mx)
    D_min = torch.min(D)
    eps = 0.009
    if D_min < eps:
        D_1 = torch.zeros_like(D)
        D_1[D > D_min] = 1 / D[D > D_min]  # .sqrt()
    else:
        D_1 = 1 / D  # .sqrt()
    # D_1 = 1 / D.clamp(min=0.005).sqrt()
    return U @ D_1.sqrt().diag() @ V.t(), U @ D_1.diag() @ V.t()


def mx_tr(mx):
    return mx.diag().sum()


def get_mgrid(sidelen, dim=2):
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[: sidelen[0], : sidelen[1]], axis=-1)[
            None, ...
        ].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[0], : sidelen[1], : sidelen[2]], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError("Not implemented for dim=%d" % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords.to(torch.float32)


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


class EdgeBlock(nn.Module):
    def __init__(self, in_, out_, dtype=torch.float32) -> None:
        super(EdgeBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_, out_, dtype=dtype),
            nn.BatchNorm1d(out_, dtype=dtype),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class IGNR(nn.Module):
    def __init__(
        self, nnodes, nfeat, nhid, ep_ratio, sinkhorn_iter, nlayers=3, **kwargs
    ):
        super().__init__()
        self.step_size = nnodes
        self.nlayers = nlayers
        self.ep_ratio = ep_ratio
        self.sinkhorn_iter = sinkhorn_iter

        self.edge_index = np.array(
            list(product(range(self.step_size), range(self.step_size)))
        ).T

        self.net0 = nn.ModuleList(
            [
                EdgeBlock(nfeat, nhid),
                EdgeBlock(nhid, nhid),
                nn.Linear(nhid, 1, dtype=torch.float32),
            ]
        )

        self.net1 = nn.ModuleList(
            [
                EdgeBlock(2, nhid),
                EdgeBlock(nhid, nhid),
                nn.Linear(nhid, 1, dtype=torch.float32),
            ]
        )

        self.P = None
        self.Lx_inv = None

        self.output = nn.Linear(nhid, 1)
        self.act = F.relu
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)

    def forward(self, c, Lx=None):
        if self.P is None:
            mx_size = Lx.size(0)
            self.P = nn.Parameter(
                torch.Tensor(mx_size, self.step_size).to(torch.float32).uniform_(0, 1)
            )  # transport plan
        self.train()
        x0 = get_mgrid(c.shape[0]).to(c.device)
        c = c[self.edge_index[0]] + c[self.edge_index[1]] / 2
        for layer in range(len(self.net0)):
            c = self.net0[layer](c)
            if layer == 0:
                x = self.net1[layer](x0)
            else:
                x = self.net1[layer](x)

            if layer != (len(self.net0) - 1):
                # use node feature to guide the graphon generating process
                x = x * c
            else:
                x = (1 - self.ep_ratio) * x + self.ep_ratio * c

        adj = x.reshape(self.step_size, self.step_size)

        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        
        if Lx is not None and self.Lx_inv is None:
            self.Lx_inv = mx_inv(Lx)
        try:
            opt_loss = self.opt_loss(adj)
        except:
            opt_loss = torch.tensor(0).to(c.device)
            
        adj_hat = adj + 1e-6 + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
        edge_weight = adj_hat.flatten()
        return adj, edge_weight, opt_loss

    def opt_loss(self, adj):
        Ly_inv_rt, Ly_inv = mx_inv_sqrt(adj)
        m = self.step_size
        P = self.P.abs()

        for _ in range(self.sinkhorn_iter):
            P = P / P.sum(dim=1, keepdim=True)
            P = P / P.sum(dim=0, keepdim=True)

        # if self.args.use_symeig:
        sqrt = torch.symeig(
            Ly_inv_rt @ self.P.t() @ self.Lx_inv @ self.P @ Ly_inv_rt, eigenvectors=True
        )
        loss = torch.abs(
            mx_tr(Ly_inv) * m - 2 * torch.sqrt(sqrt[0].clamp(min=2e-20)).sum()
        )
        return loss

    @torch.no_grad()
    def inference(self, c):
        self.eval()

        x0 = get_mgrid(c.shape[0]).to(c.device)
        c = c[self.edge_index[0]] + c[self.edge_index[1]] / 2
        for layer in range(len(self.net0)):
            c = self.net0[layer](c)
            if layer == 0:
                x = self.net1[layer](x0)
            else:
                x = self.net1[layer](x)

            if layer != (len(self.net0) - 1):
                # use node feature to guide the graphon generating process
                x = x * c
            else:
                x = (1 - self.ep_ratio) * x + self.ep_ratio * c

        adj = x.reshape(self.step_size, self.step_size)

        adj = (adj + adj.T) / 2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        
        adj_hat = adj + 1e-6 + torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
        
        return adj_hat.flatten()


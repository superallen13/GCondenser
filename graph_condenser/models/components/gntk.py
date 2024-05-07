import math
import torch
import torch.nn.functional as F


def have_nan(a):
    return torch.any(torch.isnan(a))


def have_negative(a):
    return torch.any(a < 0)


class GNTK(torch.nn.Module):
    def __init__(self, num_layers, num_mlp_layers, scale, reg_lambda=1e-6):
        super(GNTK, self).__init__()
        """
        num_layers: number of layers in the neural networks (not including the input layer)
        num_mlp_layers: number of MLP layers
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.scale = scale
        assert scale in ["uniform", "degree"]
        self.reg_lambda = reg_lambda

    def __adj(self, S, adj1, adj2):
        """
        go through one adj layer
        """
        tmp = adj1.mm(S)
        tmp = adj2.mm(tmp.transpose(0, 1)).transpose(0, 1)
        return tmp

    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = torch.sqrt(S.diag())
        tmp = diag[:, None] * diag[None, :]
        S = S / tmp
        S = torch.clamp(
            S, -0.9999, 0.9999
        )  # smooth the value so the derivative will not lead into NAN: https://discuss.pytorch.org/t/nan-gradient-for-torch-cos-torch-acos/9617
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        S = S * tmp

        return S, DS, diag

    def __next(self, S):
        """
        go through one MLP layer
        """
        S = torch.clamp(S, -0.9999, 0.9999)
        DS = (math.pi - torch.acos(S)) / math.pi
        S = (S * (math.pi - torch.acos(S)) + torch.sqrt(1 - torch.pow(S, 2))) / math.pi
        return S, DS

    def diag(self, X, A):
        """
        compute the diagonal element of GNTK
        X: feature matrix
        A: adjacency matrix
        """

        if self.scale == "uniform":
            scale_mat = 1.0
        else:
            scale_mat = 1.0 / torch.outer(A.sum(dim=1), A.sum(dim=0))

        diag_list = []
        sigma = (
            torch.mm(X, X.T) + 0.0001
        )  # for smoothness, in case the diag of sigma has zero
        sigma = scale_mat * self.__adj(sigma, A, A)

        for mlp_layer in range(self.num_mlp_layers):
            sigma, dot_sigma, diag = self.__next_diag(sigma)
            diag_list.append(diag)

        return diag_list

    def forward(self, A_S, X_S, y_S, A_T, X_T):
        """
        n: # nodes
        d, c: # of features/classes
        A_S: (n, n)
        X_S: (n, d)
        y_S: (n, c)
        A_T: (n', n')
        X_T: (n', d)

        diag_T: (m, n'), m is the MLP layers
        diag_S: (m, n)
        """
        n = A_S.size(0)
        n_prime = A_T.size(0)
        device = A_S.device

        # check if the 0.0001 is necessary.
        A_S = 0.0001 * torch.eye(A_S.size(0)).to(device) + A_S - A_S
        A_T = 0.0001 * torch.eye(A_T.size(0)).to(device) + A_T

        diag_T = torch.cat(self.diag(X_T, A_T)).view(self.num_mlp_layers, -1)
        assert diag_T.size() == (self.num_mlp_layers, n_prime), "Wrong diag_T shape"

        diag_S = torch.cat(self.diag(X_S, A_S)).view(self.num_mlp_layers, -1)
        assert diag_S.size() == (self.num_mlp_layers, n), "Wrong diag_S shape"

        """
        Computing K_SS
        """
        if self.scale == "uniform":
            scale_mat = 1.0
        else:
            scale_mat = 1.0 / torch.ger(A_S.sum(dim=1), A_S.sum(dim=0))
            assert scale_mat.size() == (n, n), "Wrong scale_mat shape"

        sigma = torch.matmul(X_S, X_S.T) + 0.0001
        tmp = torch.matmul(A_S, sigma)
        sigma = torch.matmul(tmp, A_S.T)
        assert sigma.size() == (n, n), "Wrong sigma shape"
        sigma = scale_mat * sigma
        ntk = torch.clone(sigma)

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.ger(diag_S[mlp_layer, :], diag_S[mlp_layer, :]) + 0.000001
            assert tmp.size() == (n, n), "Wrong normalization matrix shape"
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        for layer in range(1, self.num_layers):
            tmp = torch.matmul(A_S, ntk)
            ntk = torch.matmul(tmp, A_S.T)
        assert ntk.size() == (n, n), "Wrong ntk shape"
        ntk = scale_mat ** (self.num_layers - 1) * ntk
        K_SS = ntk
        assert K_SS.size() == (n, n), "Wrong K_SS shape"

        """
        Computing K_ST column by column
        """
        K_ST = torch.zeros((A_S.size(0), A_T.size(0))).to(device)
        if self.scale == "uniform":
            scale_mat = 1.0
        else:
            scale_mat = 1.0 / torch.ger(A_S.sum(dim=1), A_T.sum(dim=0))
            assert scale_mat.size() == (
                n,
                n_prime,
            ), "Wrong K_ST scale_mat shape"

        sigma = torch.matmul(X_S, X_T.T) + 0.0001
        tmp = torch.matmul(A_S, sigma)
        sigma = torch.matmul(tmp, A_T.T)
        assert sigma.size() == (n, n_prime), "Wrong K_ST sigma shape"
        sigma = scale_mat * sigma
        ntk = torch.clone(sigma)

        for mlp_layer in range(self.num_mlp_layers):
            tmp = torch.ger(diag_S[mlp_layer, :], diag_T[mlp_layer, :]) + 0.000001
            assert tmp.size() == (
                n,
                n_prime,
            ), "Wrong K_ST normalization matrix shape"
            sigma = sigma / tmp
            sigma, dot_sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma
            sigma = sigma * tmp

        for layer in range(1, self.num_layers):
            tmp = torch.matmul(A_S, ntk)
            ntk = torch.matmul(tmp, A_T.T)
        assert ntk.size() == (n, n_prime), "Wrong K_ST ntk shape"
        ntk = scale_mat ** (self.num_layers - 1) * ntk
        K_ST = ntk
        assert K_ST.size() == (n, n_prime), "K_ST shape wrong."

        """
        Prediction
        """
        KSS_reg = K_SS + self.reg_lambda * torch.trace(K_SS) / n * torch.eye(n).to(
            device
        )
        KSS_inverse_yS = torch.linalg.solve(KSS_reg, y_S)
        pred = K_ST.T.mm(KSS_inverse_yS)

        return pred, K_SS


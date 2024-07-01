import os
import math
from collections import Counter

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.linalg import eigh
import dgl
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import dataset

from graph_condenser.models.condenser import Condenser
from graph_condenser.data.datamodule import CondensedGraphDataModule
from graph_condenser.models.backbones.lightning_gnn import LightningGNNSparse


def normalize_adj(mx: sp.csr_matrix) -> sp.csr_matrix:
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv_sqrt = np.power(rowsum, -0.5)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    mx_normalized = r_mat_inv_sqrt @ mx @ r_mat_inv_sqrt
    return mx_normalized


def get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat([real_eigenvals[k1_list], real_eigenvals[k2_list]])
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]],
        dim=1,
    )
    return eigenvals, eigenvecs


def get_largest_cc(edges, num_nodes):
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    src, dst = edges
    src = src.numpy().tolist()
    dst = dst.numpy().tolist()
    edge_list = list(zip(src, dst))
    nx_graph.add_edges_from(edge_list)
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    idx_lcc = list(largest_cc)
    largest_cc_graph = nx_graph.subgraph(largest_cc)
    adj_lcc = nx.adjacency_matrix(largest_cc_graph)
    adj_norm_lcc = normalize_adj(adj_lcc)
    return idx_lcc, adj_norm_lcc, adj_lcc


def get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    idx_train_lcc = list(set(idx_train).intersection(set(idx_lcc)))
    y_full = y_full.numpy()
    if len(idx_lcc) == num_nodes:
        idx_map = idx_train
    else:
        y_train = y_full[idx_train]
        y_train_lcc = y_full[idx_train_lcc]

        y_lcc_idx = list(
            (set(range(num_nodes)) - set(idx_train)).intersection(set(idx_lcc))
        )
        y_lcc_ = y_full[y_lcc_idx]
        counter_train = Counter(y_train)
        counter_train_lcc = Counter(y_train_lcc)
        idx = np.arange(len(y_lcc_))
        for c in range(num_classes):
            num_c = counter_train[c] - counter_train_lcc[c]
            if num_c > 0:
                idx_c = list(idx[y_lcc_ == c])
                idx_c = np.array(y_lcc_idx)[idx_c]
                idx_train_lcc += list(np.random.permutation(idx_c)[:num_c])
        idx_map = [idx_lcc.index(i) for i in idx_train_lcc]

    return idx_train_lcc, idx_map


def get_eigh(laplacian_matrix):
    laplacian_matrix = laplacian_matrix.todense()
    eigenvalues, eigenvectors = eigh(laplacian_matrix)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]

    return eigenvalues, eigenvectors


def syn_eigen_init(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat([real_eigenvals[k1_list], real_eigenvals[k2_list]])
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]],
        dim=1,
    )
    return eigenvals, eigenvecs


def get_init_syn_eigenvecs(n_syn, num_classes):
    n_nodes_per_class = n_syn // num_classes
    n_nodes_last = n_syn % num_classes

    size = [n_nodes_per_class for i in range(num_classes - 1)] + (
        [n_syn - (num_classes - 1) * n_nodes_per_class]
        if n_nodes_last != 0
        else [n_nodes_per_class]
    )
    prob_same_community = 1 / num_classes
    prob_diff_community = prob_same_community / 3

    prob = [
        [prob_diff_community for i in range(num_classes)] for i in range(num_classes)
    ]
    for idx in range(num_classes):
        prob[idx][idx] = prob_same_community

    syn_graph = nx.stochastic_block_model(size, prob)
    syn_graph_adj: sp.csr_matrix = nx.adjacency_matrix(syn_graph)
    # add self-loop
    syn_graph_adj = syn_graph_adj + sp.eye(n_syn)
    syn_graph_L = normalize_adj(syn_graph_adj)
    syn_graph_L = sp.eye(n_syn) - syn_graph_L
    _, eigen_vecs = get_eigh(syn_graph_L)

    return torch.FloatTensor(eigen_vecs)


def get_subspace_covariance_matrix(
    eigenvecs: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    x_trans = eigenvecs.T @ x  # kd
    x_trans = F.normalize(input=x_trans, p=2, dim=1)
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    co_matrix = torch.bmm(
        x_trans_unsqueeze.permute(0, 2, 1), x_trans_unsqueeze
    )  # kd1 @ k1d = kdd
    return co_matrix


def get_embed_sum(eigenvals, eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = torch.diag(1 - eigenvals) @ x_trans  # kd
    embed_sum = eigenvecs @ x_trans  # nk @ kd = nd
    return embed_sum


def get_embed_mean(embed_sum, label):
    class_matrix = F.one_hot(label).float()  # nc
    class_matrix = class_matrix.T  # cn
    embed_sum = class_matrix @ embed_sum  # cd
    mean_weight = (1 / class_matrix.sum(1)).unsqueeze(-1)  # c1
    embed_mean = mean_weight * embed_sum
    embed_mean = F.normalize(input=embed_mean, p=2, dim=1)
    return embed_mean


def load_eigen(preprocess_dir):
    dataset = preprocess_dir.split("/")[-1]
    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"

    eigvals_path = os.path.join(preprocess_dir, val_file_name)
    eigvecs_path = os.path.join(preprocess_dir, vec_file_name)
    eigenvalues = np.load(eigvals_path)
    eigenvectors = np.load(eigvecs_path)

    if dataset in ["arxiv", "flickr", "reddit"]:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(preprocess_dir, val_file_name)
        eigvecs_path = os.path.join(preprocess_dir, vec_file_name)
        eigenvalues_la = np.load(eigvals_path)
        eigenvectors_la = np.load(eigvecs_path)

        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]

    return eigenvalues, eigenvectors


class GDEM(Condenser):
    def __init__(
        self,
        g: dgl.DGLGraph,
        observe_mode: str,
        budget: int,
        label_distribution: str,
        init_method: str,
        validator: torch.nn.Module,
        lr_val: float,
        wd_val: float,
        tester: torch.nn.Module,
        lr_test: float,
        wd_test: float,
        ##### For GDEM #####
        eigen_k: int,
        ratio: float,
        alpha: float,
        beta: float,
        gamma: float,
        opt_eigenvec: torch.optim.Optimizer,
        opt_feat: torch.optim.Optimizer,
        gnn: torch.nn.Module,
        lr_model: float,
        wd_model: float,
        update_epoch_adj: int,
        update_epoch_feat: int,
        preprocess_dir: str = "",
    ):
        super().__init__(
            g,
            observe_mode,
            budget,
            label_distribution,
            init_method,
            validator,
            lr_val,
            wd_val,
            tester,
            lr_test,
            wd_test,
        )
        idx_lcc, idx_train_lcc, idx_map, eigenvals_lcc, eigenvecs_lcc = self.preprocess(
            g, preprocess_dir
        )

        eigenvals_lcc = torch.FloatTensor(eigenvals_lcc)
        eigenvecs_lcc = torch.FloatTensor(eigenvecs_lcc)

        eigenvals, eigenvecs = get_syn_eigen(
            real_eigenvals=eigenvals_lcc,
            real_eigenvecs=eigenvecs_lcc,
            eigen_k=eigen_k,
            ratio=ratio,
        )
        co_x_trans_real = get_subspace_covariance_matrix(
            eigenvecs, g.ndata["feat"][idx_lcc]
        )
        embed_sum = get_embed_sum(
            eigenvals=eigenvals, eigenvecs=eigenvecs, x=g.ndata["feat"][idx_lcc]
        )
        embed_sum = embed_sum[idx_map, :]
        embed_mean_real = get_embed_mean(
            embed_sum=embed_sum, label=g.ndata["label"][idx_train_lcc]
        )

        self.eigenvals_syn = eigenvals
        self.co_x_trans_real = co_x_trans_real
        self.embed_mean_real = embed_mean_real

        self.eigenvecs_syn = nn.Parameter(
            torch.FloatTensor(budget, eigen_k), requires_grad=True
        )
        init_syn_eigenvecs = get_init_syn_eigenvecs(budget, g.num_classes)
        init_syn_eigenvecs = init_syn_eigenvecs[:, :eigen_k]
        self.eigenvecs_syn.data.copy_(init_syn_eigenvecs)

    def preprocess(self, g, preprocess_dir: str = ""):
        if preprocess_dir and os.path.exists(preprocess_dir):
            idx_lcc = np.load(f"{preprocess_dir}/idx_lcc.npy")
            idx_train_lcc = np.load(f"{preprocess_dir}/idx_train_lcc.npy")
            idx_map = np.load(f"{preprocess_dir}/idx_map.npy")
            eigenvals_lcc, eigenvecs_lcc = load_eigen(preprocess_dir)
        else:
            idx_lcc, adj_norm_lcc, _ = get_largest_cc(g.edges(), g.number_of_nodes())
            idx_train_lcc, idx_map = get_train_lcc(
                idx_lcc=idx_lcc,
                idx_train=g.ndata["train_mask"].nonzero().squeeze().numpy(),
                y_full=g.ndata["label"],
                num_nodes=g.number_of_nodes(),
                num_classes=g.num_classes,
            )
            L_lcc = sp.eye(len(idx_lcc)) - adj_norm_lcc
            eigenvals_lcc, eigenvecs_lcc = get_eigh(L_lcc)
            # save to preprocess_dir
            if preprocess_dir:
                os.makedirs(preprocess_dir, exist_ok=True)
                np.save(f"{preprocess_dir}/idx_lcc.npy", idx_lcc)
                np.save(f"{preprocess_dir}/idx_train_lcc.npy", idx_train_lcc)
                np.save(f"{preprocess_dir}/idx_map.npy", idx_map)
                np.save(f"{preprocess_dir}/eigenvalues.npy", eigenvals_lcc)
                np.save(f"{preprocess_dir}/eigenvectors.npy", eigenvecs_lcc)
        return idx_lcc, idx_train_lcc, idx_map, eigenvals_lcc, eigenvecs_lcc

    def training_step(self, data, batch_idx):
        opt_eigenvec, opt_feat = self.optimizers()

        co_x_trans_real = self.co_x_trans_real.to(self.device)
        embed_mean_real = self.embed_mean_real.to(self.device)

        label_cond = self.g_cond.ndata["label"].to(self.device)
        eigenvals_syn = self.eigenvals_syn.to(self.device)
        feat_cond = self.feat_cond  # learnable
        eigenvecs_syn = self.eigenvecs_syn  # learnable

        loss = torch.tensor(0.0).to(self.device)
        # eigenbasis match
        co_x_trans_syn = get_subspace_covariance_matrix(
            eigenvecs=eigenvecs_syn, x=feat_cond
        )  # kdd
        eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
        loss += self.hparams.alpha * eigen_match_loss

        # class loss
        embed_sum_syn = get_embed_sum(
            eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=feat_cond
        )
        embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=label_cond)  # cd
        cov_embed = embed_mean_real @ embed_mean_syn.T
        iden = torch.eye(data.num_classes).to(self.device)
        class_loss = F.mse_loss(cov_embed, iden)
        loss += self.hparams.beta * class_loss

        # orthog_norm
        orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
        iden = torch.eye(self.hparams.eigen_k).to(self.device)
        orthog_norm = F.mse_loss(orthog_syn, iden)
        loss += self.hparams.gamma * orthog_norm

        self.log(
            "train/loss",
            loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        opt_eigenvec.zero_grad()
        opt_feat.zero_grad()
        self.manual_backward(loss)

        if (
            self.current_epoch
            % (self.hparams.update_epoch_adj + self.hparams.update_epoch_feat)
            < self.hparams.update_epoch_adj
        ):
            opt_eigenvec.step()
        else:
            opt_feat.step()

    def validation_step(self, data, batch_idx):
        # NOTE: currently only support GCNSparse as the validator
        g_cond = self.g_cond
        feat_cond = self.feat_cond.detach().to("cpu")
        g_cond.ndata["feat"] = feat_cond.clone()

        eigenvecs_syn = self.eigenvecs_syn.detach().to("cpu")
        eigenvals_syn = self.eigenvals_syn
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
        adj_syn = torch.eye(self.hparams.budget) - L_syn
        g_cond.adj = adj_syn

        gnn_val = self.validator(self.num_node_features, self.num_classes)
        datamodule = CondensedGraphDataModule(g_cond, data, self.hparams.observe_mode)
        model = LightningGNNSparse(gnn_val, self.hparams.lr_val, self.hparams.wd_val)
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=200,
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, datamodule)
        self.log(
            "val/acc",
            model.val_acc_best.compute(),
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_acc_best(model.val_acc_best.compute())
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best",
            self.val_acc_best.compute(),
            batch_size=1,
            sync_dist=True,
            prog_bar=True,
        )

        # make sure the memory used in the validation step is released
        del datamodule, gnn_val, model, trainer
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()

    def test_step(self, data, batch_idx):
        # NOTE: currently only support GCNSparse as the tester
        g_cond = self.g_cond
        feat_cond = self.feat_cond.detach().to("cpu")
        g_cond.ndata["feat"] = feat_cond.clone()

        eigenvecs_syn = self.eigenvecs_syn.detach().to("cpu")
        eigenvals_syn = self.eigenvals_syn
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
        adj_syn = torch.eye(self.hparams.budget) - L_syn
        g_cond.adj = adj_syn

        datamodule = CondensedGraphDataModule(g_cond, data, self.hparams.observe_mode)
        torch.set_grad_enabled(True)  # enable gradients for the test step

        results = []
        table_cols = ["test_acc_mean", "test_acc_std"]
        for i in range(5):
            gnn_test = self.tester(self.num_node_features, self.num_classes)
            model = LightningGNNSparse(
                gnn_test, self.hparams.lr_test, self.hparams.wd_test
            )
            checkpoint = ModelCheckpoint(
                dirpath=self.test_ckpt_dir,
                monitor="val_acc",
                save_top_k=1,
                mode="max",
            )
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=200,
                log_every_n_steps=1,
                callbacks=[checkpoint],
                enable_model_summary=False,
                enable_progress_bar=False,
                logger=False,
            )
            trainer.fit(model, datamodule)
            ckpt_path = checkpoint.best_model_path
            test_acc = trainer.test(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True
            )
            results.append(test_acc[0]["test_acc"])
            table_cols.append(f"test_acc_{i}")
            os.remove(ckpt_path)

        acc_mean = sum(results) / len(results)
        acc_std = torch.std(torch.tensor(results)).item()
        table_data = [acc_mean, acc_std] + results

        if self.logger is not None:
            # assume the logger is WandbLogger
            self.logger.log_table(
                key="test/acc_table", columns=table_cols, data=[table_data]
            )

        self.log(
            "test/acc_mean",
            acc_mean,
            batch_size=1,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "test/acc_std",
            acc_std,
            batch_size=1,
            sync_dist=True,
            prog_bar=True,
        )

        return acc_mean, acc_std

    def configure_optimizers(self):
        opt_eigenvec = self.hparams.opt_eigenvec(params=[self.eigenvecs_syn])
        opt_feat = self.hparams.opt_feat(params=[self.feat_cond])
        return opt_eigenvec, opt_feat

#%%
import copy
import glob
import os
import numpy as np
import torch.nn
import torch_sparse
from ogb.linkproppred import Evaluator
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv
import torch_geometric
import scipy.sparse as ssp
import pandas as pd

import XAI
import dataLoader
from utils import validation, utils_func, utils, ainb, graph_utils
from plots import plots
from GNNexplainer import gnnexplainer, get_top_edges_edge_ig, CAM
from typing import Literal, Dict, List, Union
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt

class GCNLayer(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layer = torch.nn.Linear(c_in, c_out,bias=False)

    def old_forward(self, x, adj, mask):
        if mask is not None:
            adj  = adj*mask+torch.eye(adj.shape[0])
        adj = adj.to_dense()
        return torch.spmm(adj, self.layer(x))

    def forward(self, x : torch.Tensor, adj : torch_sparse.SparseTensor, mask = None) -> torch.Tensor:
        return torch_sparse.spmm(torch.stack(adj.coo()[:2]), adj.storage.value(), *adj.sparse_sizes(), self.layer(x))
    
    def precise_forward(self, x : torch.Tensor, adj : torch_sparse.SparseTensor, mask) -> torch.Tensor:
        if mask is not None:
            adj  = adj*mask+torch.eye(adj.shape[0])
        return torch.spmm(adj.to(torch.float64).to_dense(), self.layer(x))
    
class testGCN():
    def __init__(self, gnn):
        super().__init__()
        self.layers = [GCNLayer(s, 256) for s in (128, 256, 256)]

        self.layers[0].layer.weight.data, self.layers[1].layer.weight.data, self.layers[
        2].layer.weight.data = gnn.input.lin.weight.data, gnn.hidden.lin.weight.data, gnn.output.lin.weight.data

    def forward(self, x, adj, masks = None):
        tmp = x
        if masks is None:
            masks = [None] * len(self.layers)
        for layer, mask in zip(self.layers, masks):
            if self.layers.index(layer) < 2:
                tmp = relu(layer.forward(tmp, adj, mask))
            else :
                tmp = layer.forward(tmp, adj, mask)
        return tmp


class GNN(torch.nn.Module):
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def old__init__(self):
        # build GNN here
        super(GNN, self).__init__()
        #super().__init__()
        self.input = GCNConv(128, 256, bias=False)
        self.hidden = GCNConv(256, 256, bias=False)
        self.output = GCNConv(256, 256, bias=False)

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()
        #super().__init__()
        self.input = GCNConv(128, 256, add_self_loops= False, normalize= False, bias=False)
        self.hidden = GCNConv(256, 256, add_self_loops= False, normalize= False, bias=False)
        self.output = GCNConv(256, 256, add_self_loops= False, normalize= False, bias=False)

    def forward(self, x, edge_index, mask=None, deg=None):
        if mask is not None:
            n = x.shape[0]
            tmp = edge_index + torch.eye(edge_index.shape[0]) # edge_index is the adjacency matrix in sparse form
            deg = tmp.sum(dim=0).pow(-0.5)
            deg[deg == float('inf')] = 0
            A = deg.view(-1, 1) * tmp * deg.view(1, -1)
            edge_index_tmp = torch_sparse.SparseTensor.from_dense(edge_index*mask[0]+torch.eye(n)) # turn into sparse
            h = self.input(x, torch_sparse.SparseTensor.from_dense(edge_index))
            X = relu(h)
            edge_index_tmp = torch_sparse.SparseTensor.from_dense(edge_index * mask[1] + torch.eye(n))
            h = self.hidden(X, edge_index_tmp)
            X = relu(h)

            h = self.output(X, torch_sparse.SparseTensor.from_dense(edge_index))

        else:

            h = self.input(x, edge_index)
            X = relu(h)
            h = self.hidden(X, edge_index)
            X = relu(h)
            h = self.output(X, edge_index)
        return h

    def lrp_node(self, x, edge_index, r_src, r_tar, tar, epsilon=0, gamma=0):
        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + (gamma * torch.clamp(cp.lin.weight, min=0))
                return cp

        A = [None] * 3
        R = [None] * 4

        R[-1] = r_tar +r_src

        x.requires_grad_(True)

        A[0] = x.data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2], edge_index)
        s = R[3] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[2] = A[2].data * c

        z = epsilon + roh(self.hidden).forward(A[1], edge_index)
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[1] = A[1].data * c

        z = epsilon + roh(self.input).forward(A[0], edge_index)
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        R[0] = A[0].data * c
        # print(R[0].shape)
        return R[0].sum(dim=1).detach().numpy()

    def roh(self, layer : GCNConv, gamma : Union[float, torch.Tensor]):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + (gamma * torch.clamp(cp.lin.weight, min=0))
                return cp
            
    def refactored_lrp_step(self, h : torch.Tensor, edge_index : torch_sparse.SparseTensor, layer : GCNConv, node_relevance, node, epsilon= 0., gamma= 0., lrp_locals= None):
        """
        Thoroughly tested :) -> TODO: turn the comments into a test
        """

        simplified_ei = graph_utils.get_single_node_adjacency(edge_index, node, 'forward')
        # assert ((edge_index @ h)[node] == (simplified_ei @ h)[node]).all()
        # assert (layer(h, edge_index)[node] == layer(h, simplified_ei)[node]).all()
        # assert (self.roh(layer, gamma)(h, edge_index)[node] == self.roh(layer, gamma)(h, simplified_ei)[node]).all()
        # assert (epsilon + self.roh(layer, gamma)(h, edge_index)[node] == epsilon + self.roh(layer, gamma)(h, simplified_ei)[node]).all()
        z = epsilon + self.roh(layer, gamma).forward(h, simplified_ei)
        z = z[node]
        s = node_relevance / (z + 1e-15)
        c = torch.autograd.grad((z * s.data).sum(), h)[0]
        out_grad = h*c
        # z = epsilon + self.roh(layer, gamma).forward(h, simplified_ei)
        # z = z[node]
        # s = node_relevance / (z + 1e-15)
        # (z * s.data).sum().backward()
        # c = h.grad
        # out_backward = h*c
        # assert (out_grad == out_backward).all(), "Gradient yields different results"
        return out_grad
    
    def refactored_lrp_loop(self, x, edge_index, walk, r_src, r_tar, tar, epsilon= 0., gamma= 0., lrp_locals = None):

        A = [None] * 3
        R = [None] * 4

        x.requires_grad_(True)

        A[0] = x.data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        if walk[-1] == tar:
            R[-1] = r_tar
        else:
            R[-1] = r_src

        R[2] = self.refactored_lrp_step(A[2], edge_index, self.output, R[3], walk[3], epsilon, gamma, lrp_locals)[walk[2]]
        R[1] = self.refactored_lrp_step(A[1], edge_index, self.hidden, R[2], walk[2], epsilon, gamma, lrp_locals)[walk[1]]
        R[0] = self.refactored_lrp_step(A[0], edge_index, self.input, R[1], walk[1], epsilon, gamma, lrp_locals)[walk[0]]

        return R[3].sum().detach().numpy(), R[2].sum().detach().numpy(), R[1].sum().detach().numpy(), R[
            0].sum().detach().numpy()

    def lrp_return_locals(self, x, edge_index, walk, r_src, r_tar, tar, epsilon=0, gamma=0):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + (gamma * torch.clamp(cp.lin.weight, min=0))
                return cp

        A = [None] * 3
        R = [None] * 4
        Z = [None] * 3

        x.requires_grad_(True)

        A[0] = x.data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        if walk[-1] == tar:
            R[-1] = r_tar
        else:
            R[-1] = r_src

        z = epsilon + roh(self.output).forward(A[2], edge_index)
        Z[2] = copy.deepcopy(z.detach())
        z = z[walk[3]]
        s = R[3] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[2] = A[2].data * c
        R[2] = R[2][walk[2]]

        z = epsilon + roh(self.hidden).forward(A[1], edge_index)
        Z[1] = copy.deepcopy(z.detach())
        z = z[walk[2]]
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[1] = A[1].data * c
        R[1] = R[1][walk[1]]

        z = epsilon + roh(self.input).forward(A[0], edge_index)
        Z[0] = copy.deepcopy(z.detach())
        z = z[walk[1]]
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        R[0] = A[0].data * c
        R[0] = R[0][walk[0]]

        R = [r.detach() for r in R[::-1]]

        return copy.deepcopy({k : v for k,v in locals().items() if k in ['R', 'A', 'Z']})

    def lrp(self, x, edge_index, walk, r_src, r_tar, tar, epsilon=0, gamma=0):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + (gamma * torch.clamp(cp.lin.weight, min=0))
                return cp

        A = [None] * 3
        R = [None] * 4

        x.requires_grad_(True)

        A[0] = x.data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        if walk[-1] == tar:
            R[-1] = r_tar
        else:
            R[-1] = r_src

        z = epsilon + roh(self.output).forward(A[2], edge_index)
        z = z[walk[3]]
        s = R[3] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[2] = A[2].data * c
        R[2] = R[2][walk[2]]

        z = epsilon + roh(self.hidden).forward(A[1], edge_index)
        z = z[walk[2]]
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[1] = A[1].data * c
        R[1] = R[1][walk[1]]

        z = epsilon + roh(self.input).forward(A[0], edge_index)
        z = z[walk[1]]
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        R[0] = A[0].data * c
        R[0] = R[0][walk[0]]


        return R[3].sum().detach().numpy(), R[2].sum().detach().numpy(), R[1].sum().detach().numpy(), R[
            0].sum().detach().numpy()


class NN(torch.nn.Module):
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self):
        # build MLP here
        super(NN, self).__init__()
        self.input = torch.nn.Linear(256, 256, bias=False)
        self.hidden = torch.nn.Linear(256, 256, bias=False)
        self.output = torch.nn.Linear(256, 1, bias=False)

    def forward(self, x,src=None, tar=None, emb=None, classes=False):
        if src is not None and tar is not None:
            x = src + tar
        h = self.input(x)
        X = relu(h)
        h = self.hidden(X)
        X = relu(h)
        h = self.output(X)

        return h

    # noinspection PyTypeChecker
    def lrp(self, src, tar, r, epsilon=0, gamma=0):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.weight[:, :] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3
        R[-1] = r
        src = src.data.clone().requires_grad_(True)
        tar = tar.data.clone().requires_grad_(True)

        A[0] = src + tar
        A[0] = A[0].data.clone().requires_grad_(True)
        A[1] = relu(self.input(src + tar)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1])).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2])
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c

        z = epsilon + roh(self.hidden).forward(A[1])
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        z = epsilon + roh(self.input).forward(src + tar)
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        src_grad = src.grad
        tar_grad = tar.grad

        return src * src_grad, tar * tar_grad


def train(batchsize, train_set, x, adj, optimizer, gnn, nn):
    # generating random permutation
    permutation = torch.randperm(train_set["source_node"].shape[0])
    total_loss = []
    num_sample = 0

    for i in tqdm(range(0, train_set["source_node"].shape[0], batchsize), desc= "Batch nr: "):
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train_set["source_node"][idx], train_set["target_node"][idx]

        # removing positive link for training
        if batchsize > 100:
            tmp = adj.to_dense()
            tmp[train_src, train_tar] = 0
            tmp[train_tar, train_src] = 0
            tmp = torch_sparse.SparseTensor.from_dense(tmp)
        else:
            tmp = graph_utils.remove_connections_between(adj, train_src, train_tar)

        graph_rep = gnn(x, tmp)

        # positive sampling
        out = torch.sigmoid(nn(graph_rep[train_src], graph_rep[train_tar]))
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        neg_tar = torch.randint(low=0, high=len(graph_rep), size=train_src.size(), dtype=torch.long)  # 30657
        out = torch.sigmoid(nn(graph_rep[train_src], graph_rep[neg_tar]))
        neg_loss = - torch.mean(torch.log(1 - out + 1e-15))

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize

    return sum(total_loss) / num_sample

@torch.no_grad()
def test(batchsize, data_set, x, adj, evaluator, gnn, nn, accuracy=False):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    pos_preds, neg_preds = [], []
    #graph_rep = gnn(x, adj)
    graph_rep = gnn.forward(x, adj)
    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]

        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [torch.sigmoid(nn(graph_rep[src_tmp], graph_rep[tar_tmp]).squeeze(1).cpu())]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [torch.sigmoid(nn(graph_rep[src_tmp], graph_rep[tar_neg_tmp]).squeeze(1).cpu())]

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)

    if accuracy: plots.accuracy(pos_pred, neg_pred)

    neg_pred = neg_pred.view(-1, 20)

    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def main(batchsize=None, epochs=1, explain=True, save=False, train_model=False, load=True, runs=1, plot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.default_rng()

    # loading the data
    dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
    # dataset = dataLoader.ToyData("data/", "mini_graph", use_subset=True)

    data = dataset.load()
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]
    tmp = data.adj_t.set_diag()
    deg = tmp.sum(dim=0).pow(-0.5)
    deg[deg == float('inf')] = 0
    tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
    data.adj_t = tmp

    # initilaization models
    gnn, nn = GNN(), NN()
    if load:
        gnn.load_state_dict(torch.load("models/gnn_2100_50_0015"))
        nn.load_state_dict(torch.load("models/nn_2100_50_0015"))
        # gnn.load_state_dict(torch.load("models/gnn_new"))
        # nn.load_state_dict(torch.load("models/nn_new"))
    gnn.to(device), nn.to(device), data.to(device)
    t_GCN = testGCN(gnn)
    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(nn.parameters()), lr=0.0005)
    evaluator = Evaluator(name='ogbl-citation2')

    # adjusting batchsize for full Dataset
    if batchsize is None:
        batchsize = dataset.num_edges
    if explain:
        # to plot proper plot the the LRP-method we need all walks:
        explain_data = dataset.load(transform=False, explain=False)
        exp_adj = utils_func.adjMatrix(explain_data.edge_index,
                                       explain_data.num_nodes)  # Transpose of adj Matrix for find walks

    # ----------------------- training & testing
    average = np.zeros((runs, 2))
    for run in range(runs):
        valid_mrr, test_mrr, loss = torch.zeros((epochs, 1)), torch.zeros((epochs, 1)), torch.zeros((epochs, 1))
        best = 0
        for i in tqdm(range(epochs), desc= "Epoch: "):
            if train_model:
                loss[i] = train(batchsize, train_set, data.x, data.adj_t, optimizer, gnn, nn).detach()
            valid_mrr[i] = test(batchsize, valid_set, data.x, data.adj_t, evaluator, gnn, nn)
            test_mrr[i] = test(batchsize, test_set, data.x, data.adj_t, evaluator, gnn, nn)
            # valid_mrr[i] = test(batchsize, valid_set, data.x, data.adj_t, evaluator, t_GCN, nn)
            # test_mrr[i] = test(batchsize, test_set, data.x, data.adj_t, evaluator, t_GCN, nn)

            if valid_mrr[i] > best and save:
                best = valid_mrr[i]
                tmp_gnn = copy.deepcopy(gnn.state_dict())
                tmp_nn = copy.deepcopy(nn.state_dict())

            if i == epochs - 1:
                if save:
                    torch.save(tmp_gnn, "models/gnn_None_50_001_new")
                    torch.save(tmp_nn, "models/nn_None_50_001_new")
                    # torch.save(tmp_gnn, "models/gnn_new")
                    # torch.save(tmp_nn, "models/nn_new")
                if plot:
                    plots.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                                      ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error',
                                      file_name="GNN" + "performance")
                if explain:
                    XAI.explain_all_walks(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, remove_connections= True)
                    plots.plot_all_walks_in_folder("all_walk_relevances/", data.adj_t, save= True)
                    # XAI.refactored_explains(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, remove_connections= True)
                    # XAI.explains(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, False)
                    # XAI.get_explanations(data,explain_data,exp_adj,valid_set,t_GCN, gnn, nn,)



        average[run, 0] = valid_mrr[-1]
        average[run, 1] = test_mrr[-1]

    print("Testset avarage Performance:", average[:, 1].mean(), "Testset variance:",
          ((average[:, 1] - average[:, 1].mean()) ** 2 / runs).sum())
    print("Validation avarage Performance:", average[:, 0].mean(), "Validation variance:",
          ((average[:, 0] - average[:, 0].mean()) ** 2 / runs).sum())

#%%
if __name__ == "__main__":
    # main(None, 100, True, False, True, False, 1, False)
    main(
        batchsize=10,
        epochs= 100,
        explain= False,
        save= False,
        train_model= True,
        load= False,
        runs= 1,
        plot= False,
    )

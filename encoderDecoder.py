#%%
import copy
import numpy as np
import torch.nn
import torch_sparse
from ogb.linkproppred import Evaluator
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv
import torch_geometric
import scipy.sparse as ssp
import pandas as pd
import dataLoader
from utils import validation, utils_func, utils, ainb
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

    def forward(self, x : torch.Tensor, adj : torch_sparse.SparseTensor, mask) -> torch.Tensor:
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

    def forward(self, x, adj, masks):
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
        
        simplified_ei = get_single_node_adjacency(edge_index, node, 'forward')
        assert ((edge_index @ h)[node] == (simplified_ei @ h)[node]).all()
        assert (layer(h, edge_index)[node] == layer(h, simplified_ei)[node]).all()
        assert (self.roh(layer, gamma)(h, edge_index)[node] == self.roh(layer, gamma)(h, simplified_ei)[node]).all()
        assert (epsilon + self.roh(layer, gamma)(h, edge_index)[node] == epsilon + self.roh(layer, gamma)(h, simplified_ei)[node]).all()
        z = epsilon + self.roh(layer, gamma).forward(h, get_single_node_adjacency(edge_index, node))
        z = z[node]
        s = node_relevance / (z + 1e-15)
        c = torch.autograd.grad((z * s.data).sum(), h)[0]
        out_grad = h*c
        z = epsilon + self.roh(layer, gamma).forward(h, get_single_node_adjacency(edge_index, node))
        z = z[node]
        s = node_relevance / (z + 1e-15)
        (z * s.data).sum().backward()
        c = h.grad
        out_backward = h*c
        assert (out_grad == out_backward).all(), "Gradient yields different results"
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

        return copy.deepcopy({k : [_v.detach().sum().numpy() for _v in v] for k,v in locals().items() if k in ['R', 'A']})

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

    for i in range(0, train_set["source_node"].shape[0], batchsize):
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train_set["source_node"][idx], train_set["target_node"][idx]

        # removing positive link for training
        tmp = adj.to_dense()
        tmp[train_src, train_tar] = 0
        tmp[train_tar, train_src] = 0
        tmp = torch_sparse.SparseTensor.from_dense(tmp)
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


def explains(test_set, gnn, mlp, adj, x, edge_index,walks, validation_plot=False, prunning=True, masking=False,
             similarity=False,
             plot=True, relevances=False,
             remove_connections= False):
    src, tar = test_set["source_node"], test_set["target_node"]
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    samples = [5, 47 , 53, 5, 188, 105]
    structure = None
    random = False
    val_mul = []
    score = 0
    e = 0.0
    gammas = [0.0]  # [0.0, 0.02,0.0,0.02]
    gamma = 0.02
    z = copy.deepcopy(x)

    for gamma in gammas:
        if validation_plot: val = []
        for i in samples:
            # print(src[i], tar[i])

            p = []
            tmp = adj.clone()
            if remove_connections:
                tmp[src[i], tar[i]] = 0
                tmp[tar[i], src[i]] = 0
            walks = utils_func.walks(tmp, src[i], tar[i])

            r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i], gamma=gamma, epsilon=e)
            node_exp = gnn.lrp_node(x, edge_index, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)

            if relevances: plots.layers_sum(walks, gnn, r_src, r_tar, tar[i], x, edge_index, pos_pred[i])
            _tmp = edge_index.clone()
            if remove_connections:
                tmp = edge_index.to_dense()
            for walk in walks:
                if prunning:
                    if remove_connections:
                        _store_tmp = (tmp[src[i], tar[i]], tmp[tar[i], src[i]])
                        tmp[src[i], tar[i]] = 0
                        tmp[tar[i], src[i]] = 0
                        _tmp = torch_sparse.to_torch_sparse(tmp.nonzero().T, tmp[tmp.nonzero(as_tuple=True)].flatten(), tmp.shape[0], tmp.shape[1]).coalesce()
                        tmp[src[i], tar[i]], tmp[tar[i], src[i]] = _store_tmp
                    p.append(gnn.lrp(x, _tmp, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)[-1])


                if masking:
                    utils_func.masking(gnn, mlp, z, src[i], tar[i], edge_index, adj, walk, gamma=gamma)
            if validation_plot:
                if random:
                    p = validation.validation_random(walks, (r_src.detach().sum() + r_tar.detach().sum()))
                val.append(validation.validation_results(gnn, mlp, x, edge_index, walks, p, src[i], tar[i],
                                                         pruning=True, activaton=False))
            if similarity: score += utils_func.similarity(walks, p, x, tar[i], "max")

            if plot:
                walks.append([src[i].numpy(), src[i].numpy(), src[i].numpy(), tar[i].numpy()])
                structure = plots.plot_explain(p, src[i], tar[i], walks, "pos", gamma, structure, use_structure= False)
                #plots.plt_node_lrp(node_exp,  src[i], tar[i], walks)

        if gamma == 0.02: e = 0.2

        if validation_plot and average:
            val_mul.append(validation.validation_avg_plot(val, 57))
    if validation_plot and average:
        validation.validation_multiplot(val_mul[0], val_mul[1], val_mul[2], val_mul[3])
        validation.sumUnderCurve(val_mul[0], val_mul[2], val_mul[1], val_mul[3])
    if similarity:
        score /= len(samples)
        print("similarity score is:", score)


def refactored_explains(test_set, gnn, mlp, adj, x, edge_index,
             similarity=False,
             plot=True, relevances=False,
             remove_connections= False,
             **kwargs,
             ):

    src, tar = test_set["source_node"], test_set["target_node"]
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    samples = find_good_samples(edge_index, test_set, criterion= "indirect connections", load = "walks with indirect connections.pkl", save =  False)
    # samples = [5]#, 47 , 53, 5, 188, 105]
    # new_samples = 94

    score = 0
    e = 0.0
    gammas = [0.0]  # [0.0, 0.02,0.0,0.02]
    gamma = 0.02
    z = copy.deepcopy(x)

    for gamma in gammas:
        for i in samples:
            structure = None
            p = []
            walks, special_walks_indexes = samples[i]
            if len(walks) > 100: continue

            r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i], gamma=gamma, epsilon=e)
            node_exp = gnn.lrp_node(x, edge_index, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)

            if relevances: plots.layers_sum(walks, gnn, r_src, r_tar, tar[i], x, edge_index, pos_pred[i])
            tmp = edge_index.clone()
            for walk in walks:
                if remove_connections:
                    indexes = find_index_of_connection(edge_index, src[i], tar[i])
                    _tmp = remove_connection_at_index(tmp, indexes)
                else: _tmp = tmp
                _rel = gnn.refactored_lrp_loop(x, _tmp, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)
                p.append(_rel[-1])
                if _rel != gnn.lrp(x, _tmp, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e):
                    lrp_locals= gnn.lrp_return_locals(x, _tmp, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)
                    gnn.refactored_lrp_loop(x, _tmp, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e, lrp_locals= lrp_locals)


            if similarity: score += utils_func.similarity(walks, p, x, tar[i], "max")

            if plot:
                walks.append([src[i].numpy(), src[i].numpy(), src[i].numpy(), tar[i].numpy()])
                fig, axs = plt.subplots(1, 2, figsize=(12,6))
                structure = plots.refactored_plot_explain(p, src[i], tar[i], walks, "pos", gamma, structure, use_structure= True, ax= axs[0])
                plots.refactored_plot_explain([_p if j in special_walks_indexes else np.array(0) for j,_p in enumerate(p)], src[i], tar[i], walks, "pos", gamma, structure, use_structure= True, ax= axs[1])
                axs[1].legend().remove()
                plt.show()
                #plots.plt_node_lrp(node_exp,  src[i], tar[i], walks)

    if similarity:
        print("similarity score is:", torch.mean(score))

def find_good_samples(adj : torch_sparse.SparseTensor, subset : Dict[str, Union[torch.Tensor, List[torch.Tensor]]], criterion : Literal['indirect connections'] = None, remove_connection : bool = True, load : str = "", save : bool = False, **kwargs) -> Dict[int, List[List[int]]]:
    """
    Finds good samples to analyze based on a given criterion

    @param: load: If load is not an empty string, try "torch.load(load)". If it fails continue with criterion
    @param: save: If True, then store a pickle file with the computed walks
    """
    if load != "":
        try:
            with open(load, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError as err:
            print(f"Loadig from {load} failed")
            print(err)
            print(f"Continuing with criterion: {criterion}")
    if criterion is None: return [5, 47 , 53, 5, 188, 105]
    if criterion == "indirect connections":
        out = _find_samples_with_indirect_connections(adj, subset, remove_connection)

    if save:
        with open(f"walks with {criterion}.pkl", "wb") as f:
            pickle.dump(out, f)
    return out
    
def _find_samples_with_indirect_connections(adj : torch_sparse.SparseTensor, subset : Dict[str, torch.Tensor], remove_connection : bool = True) -> Dict[int, List[List[int]]]:
    out = {}
    for i, (src, tar, neg_tar) in tqdm(enumerate(zip(*subset.values())), desc= "Alternative connections between src and tar", total= len(subset["source_node"])):
        if remove_connection:
            _adj = remove_connection_at_index(adj, find_index_of_connection(adj, src, tar))
        else: _adj = adj
        walks = utils_func.walks(_adj, src, tar)
        indirect_connection_walks = [int(i) for i,w in enumerate(walks) if src.item() in w and tar.item() in w]
        if len(indirect_connection_walks) > 0:
            out[i] = [walks, indirect_connection_walks]
    return out

def find_index_of_connection(sparse_tensor : torch_sparse.SparseTensor, src : int, tar : int) -> torch.Tensor:
    """
    Finds the index in the sparse storage for a given connection, returns the indexes in a Tensor
    """
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    _rows = torch.where(rows == tar)[0]
    _cols = torch.where(cols == src)[0]
    _rows_t = torch.where(rows == src)[0]
    _cols_t = torch.where(cols == tar)[0]
    idx = ainb.ainb(_rows, _cols)
    idx_t = ainb.ainb(_rows_t, _cols_t)
    assert idx.sum() <= 1 and idx_t.sum() <= 1, "Should find one match at most" # In the case of repeated indexes we are most likely seeing a runtime error
    remove_indexes = []
    if idx.sum(): remove_indexes.append(_rows[idx])
    if idx_t.sum(): remove_indexes.append(_rows_t[idx_t])
    remove_indexes.sort()
    assert torch.tensor(
        [[rang[r_idx] in [src, tar] for r_idx in remove_indexes] for rang in [rows, cols]]
        ).all() #  Checks that the indexes to be removed actually correspond to src and target
    
    return remove_indexes

def _find_index_of_neighbours(sparse_tensor : torch_sparse.SparseTensor, node : Union[int, torch.Tensor], direction : Literal['forward', 'backward'] = 'backward') -> torch.Tensor:
    prev_node, next_node = sparse_tensor.storage.row(), sparse_tensor.storage.col()

    if direction == "forward":    
        return torch.where(prev_node == node)[0]
    elif direction == "backward":
        return torch.where(next_node == node)[0]
    else:
        raise ValueError("Direction not recognized")

def get_single_node_adjacency(sparse_tensor : torch_sparse.SparseTensor, node : Union[int, torch.Tensor], direction : Literal['forward', 'backward'] = 'backward') -> torch_sparse.SparseTensor:
    rows, cols, val = sparse_tensor.coo()

    if direction == "forward":    
         idx = _find_index_of_neighbours(sparse_tensor, node, 'forward')
    elif direction == "backward":
        idx = _find_index_of_neighbours(sparse_tensor, node, 'backward')
    else:
        raise ValueError("Direction not recognized")
    
    return torch_sparse.SparseTensor(row= rows[idx], col= cols[idx], value= val[idx], sparse_sizes=sparse_tensor.storage.sparse_sizes())

def remove_connection_at_index(sparse_tensor : torch_sparse.SparseTensor, indexes : list) -> torch_sparse.SparseTensor:
    """
    Removes the connection at the indexes in the sparse storage, returns a new SparseTensor without the connections
    """
    if not len(indexes) > 0: return sparse_tensor # If connection is already missing just return same tensor
    new_row, new_col, new_value = [
        torch.cat([inner_tensor[i+1: j] for i,j in zip([-1] + indexes[:-1], indexes)]) # Adding -1 at the beginning of the indexes means start from index 0
        for inner_tensor in [
                            sparse_tensor.storage.row(),
                            sparse_tensor.storage.col(),
                            sparse_tensor.storage.value()
                            ]
    ]
    return torch_sparse.SparseTensor(new_row, None, new_col, new_value, sparse_tensor.sizes())

@torch.no_grad()
def test(batchsize, data_set, x, adj, evaluator, gnn, nn, accuracy=False):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    pos_preds, neg_preds = [], []
    #graph_rep = gnn(x, adj)
    graph_rep = gnn.forward(x, adj, masks=None)
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
        for i in range(0, epochs):
            if train_model:
                loss[i] = train(batchsize, train_set, data.x, data.adj_t, optimizer, gnn, nn).detach()
            #valid_mrr[i] = test(batchsize, valid_set, data.x, data.adj_t, evaluator, gnn, nn)
            #test_mrr[i] = test(batchsize, test_set, data.x, data.adj_t, evaluator, gnn, nn)
            valid_mrr[i] = test(batchsize, valid_set, data.x, data.adj_t, evaluator, t_GCN, nn)
            test_mrr[i] = test(batchsize, test_set, data.x, data.adj_t, evaluator, t_GCN, nn)

            if valid_mrr[i] > best and save:
                best = valid_mrr[i]
                tmp_gnn = copy.deepcopy(gnn.state_dict())
                tmp_nn = copy.deepcopy(nn.state_dict())

            if i == epochs - 1:
                if save:
                    torch.save(tmp_gnn, "models/gnn_None_50_001_new")
                    torch.save(tmp_nn, "models/nn_None_50_001_new")
                if plot:
                    plots.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                                      ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error',
                                      file_name="GNN" + "performance")
                if explain:
                    refactored_explains(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, remove_connections= True)
                    explains(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, False, remove_connections= True)

                    # This generates a subgraph
                    # passable size for entries 47,188,105, 8, 10
                    src, tar = int(valid_set["source_node"][5]), int(valid_set["target_node"][5])
                    adj = data.adj_t.to_dense()
                    adj[tar,src]= 1
                    subgraph = utils_func.get_subgraph(torch_sparse.SparseTensor.from_dense(exp_adj), src, tar, 3)

                    # to do add the predicted edge back in
                    x_new, subgraph, edge, mapping = utils_func.reindex(subgraph, data.x, (src, tar))
                    tmp = torch_geometric.utils.to_dense_adj(subgraph).squeeze()
                    subgraph = torch_geometric.utils.to_dense_adj(
                        subgraph).squeeze()
                    #print(subgraph[44,116], subgraph[116,44])
                    #print(torch_sparse.SparseTensor.from_dense(subgraph))

                    explains(valid_set, gnn, nn, exp_adj, explain_data.x, data.adj_t, False)
                    walks = utils_func.walks(subgraph,edge[0],edge[1])
                    nodes = list(set([x[-1] for x in walks]))
                    mask = torch.zeros(subgraph.shape)
                    for i in nodes:
                        mask[i,i] = 1
                    walks = utils_func.map_walks(walks, mapping)
                    print(type(walks[0][0]))
                    z = gnnexplainer(subgraph.T, t_GCN, nn, edge, x_new,mask)
                    plots.plt_gnnexp(z,edge[0],edge[1], walks,mapping)

                    z = CAM(subgraph.T,gnn,x_new)
                    #z = get_top_edges_edge_ig(gnn,nn,x_new,subgraph,edge)
                    #print(torch_sparse.SparseTensor.from_dense(z))
                    plots.plot_cam(z,edge[0],edge[1],walks,mapping)

        average[run, 0] = valid_mrr[-1]
        average[run, 1] = test_mrr[-1]

    print("Testset avarage Performance:", average[:, 1].mean(), "Testset variance:",
          ((average[:, 1] - average[:, 1].mean()) ** 2 / runs).sum())
    print("Validation avarage Performance:", average[:, 0].mean(), "Validation variance:",
          ((average[:, 0] - average[:, 0].mean()) ** 2 / runs).sum())

# from importlib import reload
# reload(utils_func)
# batchsize=None; epochs=1; explain=True; save=False; train_model=False; load=True; runs=1; plot=False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# np.random.default_rng()

# # loading the data
# dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
# # dataset = dataLoader.ToyData("data/", "mini_graph", use_subset=True)

# data = dataset.load()
# split = dataset.get_edge_split()
# train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

# tmp = data.adj_t.set_diag()
# deg = tmp.sum(dim=0).pow(-0.5)
# deg[deg == float('inf')] = 0
# tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
# data.adj_t = tmp

# # initilaization models
# gnn, nn = GNN(), NN()
# if load:
#     gnn.load_state_dict(torch.load("models/gnn_2100_50_0015"))
#     nn.load_state_dict(torch.load("models/nn_2100_50_0015"))
# gnn.to(device), nn.to(device), data.to(device)
# t_GCN = testGCN(gnn)
# optimizer = torch.optim.Adam(list(gnn.parameters()) + list(nn.parameters()), lr=0.0005)
# evaluator = Evaluator(name='ogbl-citation2')

# # adjusting batchsize for full Dataset
# if batchsize is None:
#     batchsize = dataset.num_edges
# if explain:
#     # to plot proper plot the the LRP-method we need all walks:
#     explain_data = dataset.load(transform=False, explain=False)
#     exp_adj = utils_func.adjMatrix(explain_data.edge_index,
#                                     explain_data.num_nodes)  # Transpose of adj Matrix for find walks

# # ----------------------- training & testing
# average = np.zeros((runs, 2))
# #%%
# reload(utils_func)
# get_single_node_adjacency(tmp, 0, 'backward')

#%%
if __name__ == "__main__":
    # main(None, 100, True, False, True, False, 1, False)
    main()

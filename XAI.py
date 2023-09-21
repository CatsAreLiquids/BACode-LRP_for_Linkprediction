import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.nn import GCNConv
from utils import utils_func, utils
from tqdm import tqdm
import torch_sparse
import torch_geometric
import copy
import itertools
from plots import plots
from encoderDecoder import GNN, NN
from utils.graph_utils import find_index_of_connection, remove_connection_at_index, find_good_samples
from typing import Dict, List
import os



def sigm(z):
    return torch.tanh(0.5 * z) * 0.5 + 0.5


def gnnexplainer(adj, gnn, nn, edge, H0=None,mask=None, steps=10, lr=0.5, lambd=0.01, verbose=False):
    # Assumption we only use base the modified adj map
    num_layer = 3
    bar = tqdm(range(steps)) if verbose else range(steps)
    adj_t = utils_func.adj_t(adj).to_dense()
    #TODO set adj info in mask for walk between src tar
    if mask is None :
        z = (torch.ones(adj.shape) * adj * 2)
    else :
        z = mask * 2
    for i in bar:
        z.requires_grad_(True)
        emb = gnn.forward(H0, adj_t, [sigm(z)] * num_layer)
        score = nn(emb[edge[0]]+emb[edge[1]],  classes=False)  # src,tar
        emp = -score
        reg = lambd * (z ** 2).sum()  # constant
        if i in [j ** 5 for j in range(10)] and verbose: print('%5d %8.3f %8.3f' % (i, emp.item(), reg.item()))
        (emp + reg).backward()
        with torch.no_grad():
            # Check if we can set constants
            _, tmp = torch_geometric.utils.dense_to_sparse(z.grad)
            tmp = list(np.asarray(tmp.flatten()))
            tmp = set([x for x in tmp if tmp.count(x) > 1])

            if i == steps-1:
                #masking = z.grad.eq(tmp.pop())
                z = (z - lr * z.grad)
                #z[masking] = 0
            else:
                z = (z - lr * z.grad)

        z.grad = None

    return z.data

def gradCAM(adj, gnn,nn, H0):


    adj = torch_sparse.SparseTensor.from_dense(adj)

    H = gnn.forward(H0, adj,)
    H = H.sum(dim=1) / 20 ** .5

    return H

def explains_ronja(test_set, gnn, mlp, adj, x, edge_index,walks, validation_plot=False, prunning=True, masking=False,
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

def refactored_explains(
        test_set : Dict[str, torch.Tensor],
        gnn : GNN,
        mlp : NN,
        adj : object, # only for compatibility with ronja
        x : torch.Tensor,
        edge_index : torch_sparse.SparseTensor,
        similarity : bool = False,
        plot : bool = True,
        relevances : bool = False,
        remove_connections : bool = False,
        gammas : List[float] = [0.0, 0.03, 0.1],
        epsilons : List[float] = [0.0, 1e-9],
        **kwargs,
        ):

    src, tar = test_set["source_node"], test_set["target_node"]
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    sample_criterion = "indirect connections"
    samples = find_good_samples(edge_index, test_set, remove_connection= True, criterion= sample_criterion, load = "", save = "")
    # samples = [5]#, 47 , 53, 5, 188, 105]
    # new_samples = 94

    score = 0
    for epsilon in epsilons:
        for gamma in gammas:
            for i in tqdm(samples, desc= f"Good samples according to: {sample_criterion}-criterion"):
                p = []
                structure = None
                walks, special_walks_indexes = samples[i]
                if len(walks) > 1000: continue
                if remove_connections:
                    indexes = find_index_of_connection(edge_index, src[i], tar[i])
                    tmp_ei = remove_connection_at_index(edge_index.clone(), indexes)
                else: tmp_ei = edge_index

                r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i], gamma=gamma, epsilon=epsilon)
                # node_exp = gnn.lrp_node(x, tmp_ei, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)

                if relevances: plots.layers_sum(walks, gnn, r_src, r_tar, tar[i], x, tmp_ei, pos_pred[i])
                for walk in tqdm(walks, desc= "Walks in sample"):
                    _rel = gnn.refactored_lrp_loop(x, tmp_ei, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=epsilon)
                    # _old_rel = gnn.lrp(x, tmp_ei, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)
                    p.append(_rel[-1])
                    # if _rel != _old_rel:
                    #     lrp_locals= gnn.lrp_return_locals(x, tmp_ei, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)
                    #     gnn.refactored_lrp_loop(x, tmp_ei, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=e, lrp_locals= lrp_locals)


                if similarity: score += utils_func.similarity(walks, p, x, tar[i], "max")

                if plot:
                    plots.save_plots(edge_index, p, src[i], tar[i], walks, gamma, epsilon)
                    # walks.append([src[i].numpy(), src[i].numpy(), src[i].numpy(), tar[i].numpy()])
                    # structure = plots.side_by_side_plot(p, special_walks_indexes, src[i], tar[i], walks, gamma, structure)
                    #plots.plt_node_lrp(node_exp,  src[i], tar[i], walks)
                
    if similarity:
        print("similarity score is:", torch.mean(score))


def explain_all_walks(
        test_set : Dict[str, torch.Tensor],
        gnn : GNN,
        mlp : NN,
        adj : object, # only for compatibility with ronja
        x : torch.Tensor,
        edge_index : torch_sparse.SparseTensor,
        similarity : bool = False,
        plot : bool = True,
        relevances : bool = False,
        remove_connections : bool = False,
        gammas : List[float] = [0.03, 0.1],
        epsilons : List[float] = [0.0, 1e-9],
        **kwargs,
        ):
    src, tar = test_set["source_node"], test_set["target_node"]
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])

    sample_criterion = "indirect connections"
    samples = find_good_samples(edge_index, test_set, remove_connection= True, criterion= sample_criterion, load = "", save = "")
    os.makedirs("all_walk_relevances/", exist_ok= True)
    for epsilon in epsilons:
        for gamma in gammas:
            all_walks_indices = []
            all_walks_relevances = []
            for i in tqdm(samples, desc= f"Good samples according to: {sample_criterion}-criterion"):
                p = []
                walks, special_walks_indexes = samples[i]
                if len(walks) > 1000: continue
                if remove_connections:
                    indexes = find_index_of_connection(edge_index, src[i], tar[i])
                    tmp_ei = remove_connection_at_index(edge_index.clone(), indexes)
                else: tmp_ei = edge_index

                r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i], gamma=gamma, epsilon=epsilon)

                for walk in tqdm(walks, desc= "Walks in sample"):
                    p.append(gnn.refactored_lrp_loop(x, tmp_ei, walk, r_src, r_tar, tar[i], gamma=gamma, epsilon=epsilon)[-1])
                
                non_zero_inds = np.where(np.array(p) != 0)[0].tolist()
                walks_indices = torch.stack([torch.tensor(_w) for i,_w in enumerate(walks) if i in non_zero_inds]).T
                all_walks_indices.append(walks_indices)
                all_walks_relevances.extend([_p for i,_p in enumerate(p) if i in non_zero_inds])
                out = torch.sparse_coo_tensor(walks_indices, [a.item() for i,a
                in enumerate(p) if i in non_zero_inds], size =
                walks_indices.shape[0]*[edge_index.sparse_sizes()[0]])
                torch.save(out, f"all_walk_relevances/{src[i].item()}_{tar[i].item()}_{str(gamma).replace('.', ',')}_{str(epsilon).replace('.', ',')}.th")
            all_walks_indices = torch.cat(all_walks_indices, dim= 1)
            out = torch.sparse_coo_tensor(all_walks_indices, [a.item() for a in all_walks_relevances], size = walks_indices.shape[0]*[edge_index.sparse_sizes()[0]])
            torch.save(out, f"all_walk_relevances/all_samples_{str(gamma).replace('.', ',')}_{str(epsilon).replace('.', ',')}.th")

    return out




def explains(gnn, mlp, adj, x, edge_index,src,tar,walks, validation_plot=False, prunning=True, masking=False,
             similarity=False,
             plot=True, relevances=True):

    random = False
    val_mul = []
    score = 0
    epsilon = [0.0]
    gamma = [0.0]  # [0.0, 0.02,0.0,0.02]

    for e,g in itertools.product(epsilon, gamma):
        #if validation_plot: val = []
        p = []

        mid = gnn(x, edge_index)  # features, edgeindex
        pos_pred = mlp(mid[src], mid[tar])
        r_src, r_tar = mlp.lrp(mid[src], mid[tar], pos_pred, gamma=g, epsilon=e)
        #node_exp = gnn.lrp_node(x, edge_index, r_src, r_tar, tar[i], gamma=gamma, epsilon=e)

        if relevances: plots.layers_sum(walks, gnn, r_src, r_tar, tar, x, edge_index, pos_pred)
        for walk in walks:
            if prunning:
                p.append(gnn.lrp(x, edge_index, walk, r_src, r_tar, tar, gamma=g, epsilon=e)[-1])
            if masking:
                z = copy.deepcopy(x)
                utils_func.masking(gnn, mlp, z, src, tar, edge_index, adj, walk, gamma=g)
        if validation_plot:
            if random:
                p = validation.validation_random(walks, (r_src.detach().sum() + r_tar.detach().sum()))
            val.append(validation.validation_results(gnn, mlp, x, edge_index, walks, p, src[i], tar[i],
                                                         pruning=True, activaton=False))
        if similarity: score += utils_func.similarity(walks, p, x, tar[i], "max")
        if plot:
            plots.plot_explain(p, src, tar, walks, "pos", g)
            #plots.plt_node_lrp(node_exp,  src[i], tar[i], walks)

        if validation_plot and average:
            val_mul.append(validation.validation_avg_plot(val, 57))
    if validation_plot and average:
        validation.validation_multiplot(val_mul[0], val_mul[1], val_mul[2], val_mul[3])
        validation.sumUnderCurve(val_mul[0], val_mul[2], val_mul[1], val_mul[3])
    if similarity:
        score /= len(samples)
        print("similarity score is:", score)

def create_subgraph(src,tar,data,adj):

    subgraph = utils_func.get_subgraph(torch_sparse.SparseTensor.from_dense(adj), src, tar, 3)

    x_new, subgraph, edge, mapping = utils_func.reindex(subgraph, data.x, (src, tar))
    subgraph = torch_geometric.utils.to_dense_adj(
        subgraph).squeeze()

    return x_new, subgraph, edge, mapping

def get_explanations(data,exp_data,exp_adj,data_set,t_GCN, gnn, nn,  LRP=True, CAM=True, gnnexp=True,plot= True):
    samples = [47] #[47 , 53, 5, 188, 105] # 8 for debuggin only 4 nodes

    for sample in samples:
        src, tar = data_set["source_node"][sample], data_set["target_node"][sample]
        # Set connection between src tar to 0
        tmp_exp = exp_adj.clone()
        tmp_exp[src, tar] = 0

        tmp_adj = data.adj_t.to_dense().clone()
        tmp_adj[tar, src] = 0
        tmp_adj = torch_sparse.SparseTensor.from_dense(tmp_adj)

        # get subgraphs for CAM & gnnexplainer
        x_new, subgraph, edge, mapping = create_subgraph(int(src),int(tar), data, tmp_exp)
        walks = utils_func.walks(subgraph, edge[0], edge[1])

        #create mask for gnnexp
        nodes = list(set([x[-1] for x in walks]))
        mask = torch.zeros(subgraph.shape)
        for i in nodes:
            mask[i, i] = 1

        #reindex walk to create similar plot
        walks = utils_func.map_walks(walks, mapping)

        if LRP : explains(gnn, nn, tmp_exp, exp_data.x, tmp_adj, src,tar,walks)
        if CAM : rel_cam = gradCAM(subgraph.T,gnn,x_new)
        if gnnexp : rel_exp = gnnexplainer(subgraph.T, t_GCN, nn, edge, x_new, mask)
        # z = get_top_edges_edge_ig(gnn,nn,x_new,subgraph,edge) TODO check if I can make this one work

        if plot : #TODO unify plotting
            if rel_cam is not None: plots.plot_cam(rel_cam,edge[0],edge[1],walks,mapping)
            if rel_exp is not None: plots.plt_gnnexp(rel_exp,edge[0],edge[1], walks,mapping)


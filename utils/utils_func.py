import copy
import numpy as np
import torch
import torch_geometric
import itertools
from functools import reduce
import torch_sparse
from operator import itemgetter
from typing import List, Literal


def adjMatrix(edges, numNodes, selfLoops=True):
    adj = np.zeros((numNodes, numNodes))
    if selfLoops: adj += np.identity(numNodes)

    for edge in edges.T:
        adj[edge[0], edge[1]] = 1

    return torch.from_numpy(adj)


def find_walks(src, tar, walks):
    # walk 22063,22063,22063 --> artafect of model
    arr = np.asarray(walks)
    tmp = []

    for n in range(arr.shape[0]):
        if arr[n][2] == tar:
            tmp.append(arr[n])

        elif arr[n][2] == src:
            tmp.append(arr[n])

    return tmp


def get_nodes(adj, start):
    adj = adj.to_symmetric().to_dense()
    oneHop, twoHop = set(), set()

    oneHop.update(np.flatnonzero(adj[start]))
    for h1 in oneHop:
        twoHop.update(np.flatnonzero(adj[h1]))
    return list(oneHop), list(twoHop)


def masking(gnn, nn, input, src, tar, edge_index, adj, walk, gamma=0):
    x = input.data
    x.requires_grad_(True)

    M = [None] * 4
    M[0] = torch.FloatTensor(np.eye(x.shape[0])[walk[0]][:, np.newaxis])
    M[1] = torch.FloatTensor(np.eye(x.shape[0])[walk[1]][:, np.newaxis])
    M[2] = torch.FloatTensor(np.eye(x.shape[0])[walk[2]][:, np.newaxis])
    M[3] = torch.FloatTensor(np.eye(x.shape[0])[walk[3]][:, np.newaxis])
    H = [None] * 6

    y = x * M[0] + (1 - M[0]) * x.data

    H[0] = copy.deepcopy(gnn.input).forward(y, edge_index).clamp(min=0)
    H[0] = H[0] * M[1] + (1 - M[1]) * H[0].data

    H[1] = copy.deepcopy(gnn.hidden).forward(H[0], edge_index).clamp(min=0)
    H[1] = H[1] * M[2] + (1 - M[2]) * H[1].data

    H[2] = copy.deepcopy(gnn.output).forward(H[1], edge_index)
    H[2] = H[2] * M[3] + (1 - M[3]) * H[2].data

    x_n = H[2][src] + H[2][tar]
    H[3] = copy.deepcopy(nn.input).forward(x_n).clamp(min=0)
    H[4] = copy.deepcopy(nn.hidden).forward(H[3]).clamp(min=0)
    H[5] = copy.deepcopy(nn.output).forward(H[4])
    H[5].backward()

    return x.grad * x.data


def walks(A, src, tar, max_len = 4):
    if type(A) is torch.Tensor and A.shape[0] == A.shape[1]: return old_walks(A, src, tar)
    else:
        return refactored_walks(A, src, tar, max_len)

def refactored_walks(A : torch_sparse.SparseTensor, src : int, tar : int, max_len = 4) -> List[List[int]]:
    walks_src = [[src.item()]]
    walks_tar = [[tar.item()]]
    for _ in range(max_len - 1):
        walks_src = reduce(lambda x,y: x+y, [_get_next_step_in_walk(A, walk) for walk in walks_src])
        walks_tar = reduce(lambda x,y: x+y, [_get_next_step_in_walk(A, walk) for walk in walks_tar])
    
    out = [walk for walk in walks_src + walks_tar if len(walk) == max_len]
    return sorted(out, key=itemgetter(*range(max_len)))

def _get_next_step_in_walk(A : torch_sparse.SparseTensor, walk : List[int], direction : Literal['forward', 'backward'] = 'backward') -> List[List[int]]:
    prev_node, next_node = A.storage.col(), A.storage.row()

    if direction == "forward":    
        continuation_nodes = next_node[prev_node == walk[0]]
    elif direction == "backward":
        continuation_nodes = prev_node[next_node == walk[0]]
    else:
        raise ValueError("Direction not recognized")
    
    if not len(continuation_nodes) > 0: return [walk] # Walk doesn't continue.
    return [[continues_with.item()] + walk for continues_with in continuation_nodes]


def old_walks(A, src, tar):
    w = []
    for v1 in [src, tar]:
        for v2 in np.where(A[:, v1])[0]:
            for v3 in np.where(A[:, v2])[0]:
                for v4 in np.where(A[:, v3])[0]:
                    w += [[v4, v3, v2, v1.numpy().flatten()[0]]]
    w.sort(key=itemgetter(0,1,2,3))
    return w


def self_loops(rx, ry):
    loops = []
    scale_x = 0.1 * np.cos(np.linspace(0, 2 * np.pi, 50))
    scale_y = 0.1 * np.sin(np.linspace(0, 2 * np.pi, 50))
    if (rx[0] == rx[1] and ry[0] == ry[1]) and (rx[2] == rx[3] and ry[2] == ry[3]):
        rx = rx[0] + scale_x
        ry = ry[0] + scale_y
        loops.append((rx, ry))
        rx = rx[3] + scale_x
        ry = ry[3] + scale_y
        loops.append((rx, ry))
    elif rx[0] == rx[1] == rx[2] == rx[3] and ry[0] == ry[1] == ry[2] == ry[3]:  # Added numpy.all()
        rx = rx[0] + scale_x
        ry = ry[0] + scale_y
        loops.append((rx, ry))

    elif (rx[0] == rx[1] == rx[2] and ry[0] == ry[1] == ry[2]) or \
            (rx[0] == rx[1] and ry[0] == ry[1]):
        rx = rx[0] + scale_x
        ry = ry[0] + scale_y
        loops.append((rx, ry))

    elif (rx[1] == rx[2] == rx[3] and ry[1] == ry[2] == ry[3]) or \
            (rx[2] == rx[3] and ry[2] == ry[3]):
        rx = rx[3] + scale_x
        ry = ry[3] + scale_y
        loops.append((rx, ry))

    return loops


def similarity(walks, relevances, x, target, type):
    relevances = np.asarray(relevances).sum(axis=1)

    nodes = []
    for i in range(5):
        if type == "rand":
            idx = np.random.randint(x.shape[0] - 1)
            nodes.append(idx)
        else:
            if type == "max":
                idx = relevances.argmax()
            elif type == "min":
                idx = relevances.argmin()
            else:
                idx = np.random.randint(len(relevances) - 1)
            nodes.append(walks[idx])

            relevances = relevances.tolist()
            relevances.pop(idx)
            walks.pop(idx)
            relevances = np.asarray(relevances).flatten()

    nodes = set(np.asarray(nodes).flatten().tolist())
    score = 0
    for i in nodes:
        score += 1 - scipy.spatial.distance.cosine(x[target], x[i])

    score /= len(nodes)
    return score


def get_subgraph(adj, src, tar, hops):
    edge = torch_geometric.utils.to_edge_index(adj)

    tmp = []
    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
        src, hops, edge[0], relabel_nodes=False, directed=True, flow="target_to_source")
    tmp += edge_index.T.tolist()

    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
        src, hops, edge[0], relabel_nodes=False, directed=True, flow="source_to_target")
    tmp += edge_index.T.tolist()

    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
        tar, hops, edge[0], relabel_nodes=False, directed=True, flow="target_to_source")
    tmp += edge_index.T.tolist()

    subset, edge_index, mapping, edge_mask = torch_geometric.utils.k_hop_subgraph(
        tar, hops, edge[0], relabel_nodes=False, directed=True, flow="source_to_target")
    tmp += edge_index.T.tolist()
    return torch.asarray(list(tmp for tmp, _ in itertools.groupby(tmp))).T


def reindex(subgraph, x, edge):
    n = 0
    idxs = list(set(subgraph[0].tolist()).union(set(subgraph[1].tolist())))
    mapping = []

    for idx in idxs:
        tmp1 = np.flatnonzero(idx == subgraph[0])
        tmp2 = np.flatnonzero(idx == subgraph[1])

        # reindexing
        subgraph[0, tmp1] = n
        subgraph[1, tmp2] = n
        mapping.append(idx)
        n += 1

    return x[mapping], subgraph, (torch.tensor(mapping.index(edge[0])), torch.tensor(mapping.index(edge[1]))), mapping


def adj_t(adj):
    tmp = torch_sparse.SparseTensor.from_dense(adj).set_diag()
    deg = tmp.sum(dim=0).pow(-0.5)
    deg[deg == float('inf')] = 0
    return deg.view(-1, 1) * tmp * deg.view(1, -1)

def map_walks(walks, mapping):
    tmp = np.asarray(walks).flatten()
    new = np.zeros((tmp.shape))

    nodes = list(set(np.asarray(walks).flatten()))
    for i in nodes:
        new[np.flatnonzero(tmp == i)] = int(mapping[i])

    return new.reshape((len(walks),4)).astype(int)
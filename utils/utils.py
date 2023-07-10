import importlib
import igraph
import numpy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
import torch
import torch_sparse
from tqdm import tqdm
import scipy.sparse as ssp
from torch_geometric.data import Data
from matplotlib.lines import Line2D
"""
Originaly from : https://git.tu-berlin.de/thomas_schnake/demo_gnn_lrp
"""

def shrink(rx, ry):
    rx = numpy.array(rx)
    ry = numpy.array(ry)


    rx = 0.75 * rx + 0.25 * rx.mean()
    ry = 0.75 * ry + 0.25 * ry.mean()

    rx = numpy.concatenate([
        numpy.linspace(rx[0], rx[0], 41),
        numpy.linspace(rx[0], rx[1], 20),
        numpy.linspace(rx[1], rx[2], 20),
        numpy.linspace(rx[2], rx[2], 41), ])
    ry = numpy.concatenate([
        numpy.linspace(ry[0], ry[0], 41),
        numpy.linspace(ry[0], ry[1], 20),
        numpy.linspace(ry[1], ry[2], 20),
        numpy.linspace(ry[2], ry[2], 41)])


    filt = numpy.exp(-numpy.linspace(-2, 2, 41) ** 2)
    filt = filt / filt.sum()

    rx = numpy.convolve(rx, filt, mode='valid')
    ry = numpy.convolve(ry, filt, mode='valid')

    return rx, ry


def walks(A):
    w = []

    for v1 in numpy.arange(len(A)):
        for v2 in numpy.where(A[v1])[0]:
            for v3 in numpy.where(A[v2])[0]:
                w += [(v1, v2, v3)]

    return w


def layout(A):
    graph = igraph.Graph()
    graph.add_vertices(len(A))
    graph.add_edges(zip(*numpy.where(A == 1)))
    return numpy.array(list(graph.layout_kamada_kawai()))

def k_hop_subgraph_():
    src, tar = int(valid_set["source_node"][53]), int(valid_set["target_node"][53])
    subgraph = utils_func.get_subgraph(torch_sparse.SparseTensor.from_dense(exp_adj), src, tar, 3)
    # to do add the predicted edge back in
    x_new, subgraph, edge = utils_func.reindex(subgraph, data.x, (src, tar))
    pd.DataFrame(x_new.numpy()).to_csv("data/subgraph_" + str(edge[0]) + "_" + str(edge[1]) + "_features.csv",
                                       index=False)
    pd.DataFrame(subgraph.numpy()).to_csv("data/subgraph_" + str(edge[0]) + "_" + str(edge[1]) + "_edges.csv",
                                          index=False)


def subgraph_bfs(start, adj, k):
    connected = adj[:,start]
    src,tar = [start]* np.count_nonzero(connected), np.nonzero(connected)[0].tolist()
    subgraph_adj = np.zeros(adj.shape)
    subgraph_adj[:,start] += connected

    for i in range(0,k):
        tmp = connected*adj
        subgraph_adj += tmp
        connected = tmp.max(axis=1)
        src += np.nonzero(tmp)[1].tolist()
        tar +=np.nonzero(tmp)[0].tolist()

    return np.vstack((src,tar)), subgraph_adj


def subgraph_bfs2(start, adj, k):
    adj[start,start] = 1
    connected =  adj.max(axis=1)
    print(connected)
    src, tar = [start] * np.count_nonzero(connected), np.nonzero(connected)[0].tolist()
    subgraph_adj = np.zeros(adj.shape)
    subgraph_adj[:, start] += connected

    for i in range(0, k):
        tmp = connected * adj
        print(connected*adj.T)
        subgraph_adj += tmp
        connected = tmp.max(axis=1)
        src += np.nonzero(tmp)[1].tolist()
        tar += np.nonzero(tmp)[0].tolist()

    return np.vstack((src, tar)), subgraph_adj


def crop(adj,edge_index,start) :
    print(adj)
    num_rows = (adj !=0).sum(1)
    print(num_rows)
    r = adj.any(1)
    if r.any():
        m, n = adj.shape
        c = adj.any(0)
        out = adj[r.argmax():m - r[::-1].argmax(), c.argmax():n - c[::-1].argmax()]
    else:
        out = np.empty((0, 0), dtype=bool)
    return out

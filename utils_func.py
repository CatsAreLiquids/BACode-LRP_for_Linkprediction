import copy
import numpy as np
import torch


def adjMatrix(edges, numNodes, selfLoops=True):
    adj = np.zeros((numNodes, numNodes))

    if selfLoops: adj += np.identity(numNodes)

    for edge in edges.T:
        adj[edge[0], edge[1]] += 1

    return adj


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


def walks(A, src, tar):
    w = []
    for v1 in [src, tar]:
        for v2 in np.where(A[:, v1])[0]:
            for v3 in np.where(A[:, v2])[0]:
                for v4 in np.where(A[:, v3])[0]:
                    w += [[v4, v3, v2, v1.numpy().flatten()[0]]]

    return w


def self_loops(rx, ry):
    loops = []
    scale_x = 0.1 * numpy.cos(numpy.linspace(0, 2 * numpy.pi, 50))
    scale_y = 0.1 * numpy.sin(numpy.linspace(0, 2 * numpy.pi, 50))
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

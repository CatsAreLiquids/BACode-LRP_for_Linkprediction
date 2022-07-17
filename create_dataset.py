import PIL.TiffImagePlugin
import pandas as pd
import numpy as np
import torch
import torch_sparse
import dataLoader
from matplotlib import pyplot as plt
from collections import Counter
import copy
from numpy.random import default_rng
import utils
import utils_func
import model


def random_walk(data, start):
    tmp = data.adj_t.set_diag()
    tmp = (tmp / torch.unsqueeze(tmp.sum(dim=1), 1))

    x = torch.zeros((1, data.num_nodes))
    x[0, start] = 1
    x = torch_sparse.SparseTensor.from_dense(x)

    for i in range(5):
        x = torch_sparse.matmul(x, tmp)

    x = x.to_dense().numpy().flatten()
    x = np.around(x, decimals=5)

    helper = np.vstack((x, np.arange(0, data.num_nodes))).T

    helper = np.asarray([helper[i, 1] for i in range(helper.shape[0]) if helper[i, 0] != 0])
    pd.DataFrame(helper).to_csv("data/mini_graph", index=False)

    counter = Counter(x).items()
    keys, items = np.asarray([x[0] for x in counter]), np.asarray([x[1] + 1 for x in counter])

    tmp = np.vstack((keys, items)).T
    tmp = tmp[tmp[:, 0].argsort()]
    tmp = tmp[::-1]
    y_pos = np.arange(len(keys))

    # Plot 1 amounts of probability
    fig, axs = plt.subplots()
    axs.set_title("Amount of Probabilitys for Random Walk initiated on Paper iD : " + str(start))
    axs.set_ylabel("occurence of probaility")
    axs.set_xlabel("Probabilitys sorted desending by value")

    axs.bar(y_pos, tmp[:, 1], log=True)

    plt.setp(axs.get_xticklabels(), visible=False)
    plt.savefig("plots/bar_randomwalk_amount" + str(start))
    plt.show()

    # PLot 2 probabilitys
    fig, axs = plt.subplots()
    axs.set_title("Probabilitys of Random Walk rounded, initiated on Paper iD : " + str(start))
    axs.set_ylabel("Probaility")
    axs.set_xlabel("Probabilitys sorted desending by value")

    axs.bar(y_pos, tmp[:, 0], log=True)

    plt.setp(axs.get_xticklabels(), visible=False)
    plt.savefig("plots/bar_randomwalk_value" + str(start))
    plt.show()


def get_graph(data, year):
    idxs = pd.read_csv("data/mini_graph").to_numpy()
    idxs = idxs[:, 1]
    edges = data.edge_index.numpy().T
    res = set()

    for idx in idxs:
        res.update(np.flatnonzero(idx == edges[:, 0]).tolist())
        res.update(np.flatnonzero(idx == edges[:, 1]).tolist())
    res = np.array(list(res))

    edges = edges[res]

    idxs = set()
    for edge in edges:
        idxs.add(edge[0])
        idxs.add(edge[1])

    tmp = copy.deepcopy(idxs)
    discard = []

    for idx in idxs:
        if np.count_nonzero(idx == edges[:, 0]) + np.count_nonzero(idx == edges[:, 1]) < 3:
            discard.append(idx)

    for idx in discard:
        edges = np.delete(edges, np.flatnonzero(idx == edges[:, 0]), axis=0)
        edges = np.delete(edges, np.flatnonzero(idx == edges[:, 1]), axis=0)
        tmp.discard(idx)

    idxs = np.array(list(tmp))
    pd.DataFrame(edges).to_csv("data/mini_graph_edges")
    pd.DataFrame(idxs).to_csv("data/mini_graph_node_index")
    pd.DataFrame(data.x[idxs].numpy()).to_csv("data/mini_graph_features", index=False)
    pd.DataFrame(year[idxs]).to_csv("data/mini_graph_year", index=False)


def reindexing():
    edges = pd.read_csv("data/mini_graph_edges").to_numpy()
    idxs = pd.read_csv("data/mini_graph_node_index").to_numpy()
    edges = edges[:, 1:]
    idxs = idxs[:, 1:]
    n = 0
    for idx in idxs:
        tmp1 = np.flatnonzero(idx == edges[:, 0])
        tmp2 = np.flatnonzero(idx == edges[:, 1])

        # reindexing
        edges[tmp1, 0] = n
        edges[tmp2, 1] = n
        n += 1

    pd.DataFrame(edges).to_csv("data/mini_graph_edges_indexed", index=False)


def graph_split(data):
    year = pd.read_csv("data/mini_graph_year").to_numpy().flatten()
    edges = pd.read_csv("data/mini_graph_edges_indexed").to_numpy()
    idxs = pd.read_csv("data/mini_graph_node_index").to_numpy()

    helper = np.vstack((idxs[:, 0], year)).T
    adj = data.adj_t.to_dense()
    candidates = np.array([x[0] for x in helper if x[1] >= 2018])
    candidates = [x for x in candidates if np.count_nonzero(adj[:, x]) >= 3]
    train = copy.deepcopy(edges)

    # we aim for 99/1/1 split with 252068 Edges that means we want : 2520
    # some how tht is not really true we us 2 * 788 --> 1576 ~ 0,63%
    rng = default_rng()
    valid, test, = [], []
    neg_valid, neg_test = [], []
    for node in candidates:
        idx = np.flatnonzero(node == train[:, 0]).tolist()  # all instances of node citing
        v, t = rng.choice(idx, 2)
        valid.append(edges[v])
        test.append(edges[t])

        neg_tmp = [i[0] for i in edges if i[0] not in train[idx]]
        neg_sample = rng.choice(neg_tmp, 20)  # 20 is arbitrary for testing right now
        neg_valid.append(neg_sample)
        neg_sample = rng.choice(neg_tmp, 20)  # 20 is arbitrary for testing right now
        neg_test.append(neg_sample)

        train = np.delete(train, v, axis=0)
        train = np.delete(train, t, axis=0)

    valid_dict = {"source_node": np.array(valid)[:, 0], "target_node": np.array(valid)[:, 1],
                  "target_node_neg": neg_valid}
    test_dict = {"source_node": np.array(test)[:, 0], "target_node": np.array(test)[:, 1], "target_node_neg": neg_test}
    train_dict = {"source_node": train[:, 0], "target_node": train[:, 1]}
    torch.save(valid_dict, "data/mini_graph_valid.pt")
    torch.save(test_dict, "data/mini_graph_test.pt")
    torch.save(train_dict, "data/mini_graph_train.pt")

    # train does not include validation or test set
    pd.DataFrame(train).to_csv("data/mini_graph_edges_train", index=False)


def graph_statistics(data):
    adj = data.adj_t
    in_deg = adj.sum(dim=0)
    out_deg = adj.sum(dim=1)
    print(out_deg.sum(), in_deg.sum())
    deg = in_deg + out_deg

    avrg_in = in_deg.sum() / data.num_nodes
    avrg_out = out_deg.sum() / data.num_nodes
    avrg_deg = deg.sum() / data.num_nodes

    var_in = ((in_deg - avrg_in) ** 2).sum() / data.num_nodes
    var_out = ((out_deg - avrg_out) ** 2).sum() / data.num_nodes
    var_deg = ((deg - avrg_deg) ** 2).sum() / data.num_nodes

    max_in, max_out = max(in_deg), max(out_deg)
    min_in, min_out = min(in_deg), min(out_deg)
    max_deg, min_deg = max(deg), min(deg)

    print("Averages:    in: ", avrg_in, " out: ", avrg_out, "combined: ", avrg_deg)
    print("Variance:    in: ", var_in, " out: ", var_out, "combined: ", var_deg)
    print("Max:    in: ", max_in, " out: ", max_out, "combined: ", max_deg)
    print("Min:    in: ", min_in, " out: ", min_out, "combined: ", min_deg)

    in_deg, out_deg, deg = in_deg.numpy(), out_deg.numpy(), deg.numpy()
    print("in", np.where(np.isclose(max_in, in_deg)))
    print("out", np.where(np.isclose(max_out, out_deg)))
    print("deg", np.where(np.isclose(max_deg, deg)))
    print("Nodes: ", data.num_nodes, "Edges: ", data.num_edges, "ration: ", data.num_edges / data.num_nodes)


def create_data_baseline(adj, data):
    adj -= np.identity(data.shape[0], dtype=float)
    save = np.zeros([data.shape[0], 256])

    for i in range(adj.shape[0]):
        s1, s2 = set(), set()
        s1.update(np.where(adj[i])[0])

        for node in s1:
            s2.update(np.where(adj[node])[0])
        s2.difference_update(s1)
        s2.discard(i)
        save[i, 128:256] = data[list(s2)].sum(axis=0)
        save[i, 0:128] = data[list(s1)].sum(axis=0)
    print("finished")
    # pd.DataFrame(save).to_csv("data/baseline_NN", index=False)


def main():
    """
    Setup to compute random walk and generate Dataset from it
        dataset = dataLoader.LinkPredData("data/", use_small=False)
        data = dataset.load(transform=True)
        year = dataset.get_year()
        start = 1454242  # Any start node for the random walk, in this case the most citing paper
        random_walk(data, start)

        data = dataset.load(transform=False) # need the edge index
        get_graph(data,year)
        reindexing()

    """
    dataset = dataLoader.LinkPredData("data/", "big_graph", use_subset=False)
    data = dataset.load(transform=True, explain=False)
    graph_statistics(data)


if __name__ == "__main__":
    main()

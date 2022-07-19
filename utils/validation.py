import torch_sparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def sumUnderCurve(w, x, y, z):
    """
    creates area under curve bar plot
    """
    arr0 = np.asarray(
        [np.asarray(w).sum(), np.asarray(x).sum()])
    arr02 = np.asarray(
        [np.asarray(y).sum(), np.asarray(z).sum()])
    labels = [r"$\gamma = 0$", r"$\gamma = 0.02$"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    width = 0.35

    ax.bar(x - width / 2, arr0, 0.35, label=r"$\epsilon=0.0$", color="mediumslateblue")
    ax.bar(x + width / 2, arr02, 0.35, label=r"$\epsilon=0.2$", color="plum")
    ax.set_ylabel(r"$\sum f(x)$")
    ax.set_xticks(x, labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()

    plt.savefig("validation/lrp_sum_0_pru.pdf")
    plt.show()


def validation_multiplot(w, x, y, z):
    """
    plots mulitible validation plots (usally used for averages) in one graph
    """
    fig, axs = plt.subplots()
    l = len(lrp)
    plt.plot(np.arange(0, l, 1), w, color="plum", linestyle="--",
             label=r"$\gamma = 0.0, \epsilon= 0.0$")
    plt.plot(np.arange(0, l, 1), x, color="mediumslateblue", linestyle="-.",
             label=r"$\gamma = 0.02, \epsilon= 0.0$")
    plt.plot(np.arange(0, l, 1), y, color="palevioletred", linestyle="--",
             label=r"$\gamma = 0.0, \epsilon= 0.2$")
    plt.plot(np.arange(0, l, 1), z, color="mediumseagreen", linestyle=":",
             label=r"$\gamma = 0.02, \epsilon= 0.2$")

    tick_max = 5 * round(float(l) / 5)
    ticks = np.arange(0, tick_max + 1, 5)

    axs.set_xticks(ticks, labels=ticks)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    plt.savefig("validation/pru_multi_02.svg")
    plt.show()


def validation_avg_plot(relevances: list, l):
    """
    Creates an averaged plot over x samples
    """
    avg = np.zeros((1, l))
    for i in relevances:
        tmp = np.zeros((1, l))
        tmp[0, 0:i[1]] = i[0]
        tmp[0, i[1]:] = i[0][-1]
        avg += tmp

    avg = avg / len(relevances)
    print("value: ", avg.sum())
    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, l, 1), avg.flatten(), color="mediumslateblue")
    tick_max = 5 * round(float(l) / 5)

    ticks = np.arange(0, tick_max + 1, 5)
    axs.set_xticks(ticks, labels=ticks)
    plt.tight_layout()
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    # plt.savefig("plots/validation_pru_new_avg")
    plt.show()
    return avg.flatten()


def validation_plot(relevances: list, node, l):
    """
    Plots a single validation plot for one sample
    """

    relevances = np.asarray(relevances).flatten()

    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, l, 1), relevances, color="mediumslateblue")
    tick_max = 5 * round(float(l) / 5)
    ticks = np.arange(0, tick_max + 1, 5)
    axs.set_xticks(ticks, labels=ticks)
    axs.set_ylabel(r"$\sum f(x)$")
    plt.tight_layout()
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    plt.savefig("validation/validation_pru_rand" + str(node.numpy()) + ".svg")

    plt.show()


def restore_edges(adj, nodes, new_node, adj_new):
    adj = adj.to_dense()
    adj_new = adj_new.to_dense()

    for node in nodes:
        adj_new[node, new_node] = adj[node, new_node]
        adj_new[new_node, node] = adj[new_node, node]

    adj_new = torch_sparse.SparseTensor.from_dense(adj_new)
    return adj_new


def remove_edges(adj_new, nodes, new_node):
    adj_new = adj_new.to_dense()

    for node in nodes:
        adj_new[node, new_node] = 0
        adj_new[new_node, node] = 0

    adj_new = torch_sparse.SparseTensor.from_dense(adj_new)
    return adj_new


def clear_edges(adj, walks):
    adj_tmp = adj.to_dense()
    nodes = list(set(np.asarray(walks).flatten()))
    for node in nodes:
        for other in nodes:
            adj_tmp[node, other] = 0
            adj_tmp[other, node] = 0

    adj_tmp = torch_sparse.SparseTensor.from_dense(adj_tmp)
    return adj_tmp


def validation_list(walks: list, relevance: list, pruning: bool, activation_bool: bool):
    nodes = list(set(np.asarray(walks).flatten()))

    if activation_bool:
        r_g = 0
        activation = []
        for i in range(len(nodes)):
            r_max, old = (0, 0), -np.infty
            for node in nodes:
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node in walks[x]])
                if (s.sum() + r_g) > old:
                    r_max = (node, s.sum())
                    old = s.sum() + r_g
            nodes.remove(r_max[0])
            activation.append(r_max[0])
            r_g += r_max[1]
        return activation

    elif pruning:
        r_g = np.asarray(relevance).sum().sum()
        res_pruning = []
        for i in range(len(nodes)):
            r_min, old = (0, 0), np.infty
            for node in nodes:
                s = np.asarray([relevance[x].sum() for x in range(len(walks)) if node not in walks[x]])
                if np.abs(s.sum() - r_g) < old:
                    r_min = (node, s.sum())
                    old = np.abs(s.sum() - r_g)

            nodes.remove(r_min[0])
            res_pruning.append(r_min[0])

    return res_pruning


def validation_random(walks, out):
    l = len(walks)
    rand = np.random.uniform(-out, out, [l])
    s = rand.sum()
    return (rand / s) * out.numpy()


def validation_results(gnn, mlp, x, adj, walks, relevances, src, tar, pruning=False, activaton=False, plot=True):
    node_list = validation_list(walks, relevances, pruning=pruning, activation_bool=activaton)
    if pruning:
        adj_tmp = adj
    else:
        adj_tmp = clear_edges(adj, walks)

    mid = gnn(x, adj_tmp)
    ref = mlp(mid[src], mid[tar]).detach().numpy().sum()
    graph = []
    predictions = []
    for node in node_list:
        graph.append(node)
        if pruning:
            adj_tmp = remove_edges(adj_tmp, graph, node)
        else:
            adj_tmp = restore_edges(adj, graph, node, adj_tmp)
        mid = gnn(x, adj_tmp)
        out = mlp(mid[src], mid[tar]).detach().numpy()

        if pruning:
            predictions.append(np.abs(ref - out.sum()))
        else:
            predictions.append(out.sum() - ref)

    if plot: validation_plot(predictions, src, len(node_list))
    return predictions, len(node_list)

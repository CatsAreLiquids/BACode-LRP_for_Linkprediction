import numpy as np
import matplotlib.pyplot as plt
import igraph
import torch
import torch_geometric.utils
from openTSNE import TSNE
import scipy.sparse as ssp
from utils import utils_func, utils

def node_plt(walks, gnn, r_src, r_tar, tar, x, edge_index, pred):
    pass

def layers_sum(walks, gnn, r_src, r_tar, tar, x, edge_index, pred):
    arr = np.zeros((5, 1))
    arr[0] = pred.detach().sum()
    walks = np.asarray(walks)
    l = set(walks[:, 3])

    for node in l:
        res = gnn.lrp(x, edge_index, [node, node, node, node], r_src, r_tar, tar)
        arr[1] += res[0]
    l = set([tuple((walks[x, 2], walks[x, 3])) for x in range(walks.shape[0])])
    for node in l:
        res = gnn.lrp(x, edge_index, [node[0], node[0], node[0], node[1]], r_src, r_tar, tar)
        arr[2] += res[1]
    l = set([tuple((walks[x, 1], walks[x, 2], walks[x, 3])) for x in range(walks.shape[0])])
    for node in l:
        res = gnn.lrp(x, edge_index, [node[0], node[0], node[1], node[2]], r_src, r_tar, tar)
        arr[3] += res[2]
    for walk in walks:
        res = gnn.lrp(x, edge_index, walk, r_src, r_tar, tar)
        arr[4] += res[3]
    #print(walks)
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2, 3, 4], arr.flatten().T, width=0.35, color="mediumslateblue")
    ax.set_xticks([0, 1, 2, 3, 4],
                  labels=["f(x)", r"$\sum R_J$", r"$\sum R_{JK}$", r"$\sum R_{JKL}$", r"$\sum R_{JKLM}$"])
    ax.set_yticks([0.0, 0.225, 0.45])
    ax.set_ylabel(r"$\sum f(x)$")
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig("plots/RelevanceAtDifLayers.pdf")
    plt.show()


def plot_abs(relevances, samples):
    x_pos = np.arange(len(relevances))
    width = 0.35
    print(relevances)
    fig, ax = plt.subplots()
    ax.bar(x_pos, relevances, width, color="mediumslateblue")
    ax.set_yticks([0.0, 0.75, 1.5])
    ax.set_xticks(x_pos, labels=samples)
    ax.set_ylabel(r"$\sum f(x)$")
    plt.savefig("plots/abs_r.jpg")
    plt.show()


def baseline_lrp(R, sample):
    R = R.detach().numpy()
    keys = ['s2', 's1', 'src', 'tar', 't1', 't2']
    relevances = [R[0:128].sum(), R[128:256].sum(), R[256:384].sum(), R[382:512].sum(), R[512:640].sum(),
                  R[640:768].sum()]
    width = 0.35
    ind = np.arange(len(relevances))

    fig, ax = plt.subplots()
    for i in range(len(relevances)):
        if relevances[i] < 0:
            c = 'b'
        else:
            c = 'r'
        ax.bar(ind[i], relevances[i], width, color=c)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Relevance')
    ax.set_title('Relevance per vector')
    ax.set_xticks(ind, labels=keys)

    plt.savefig("plots/barplot_" + str(sample) + ".png")
    plt.show()


def plot_curves(epochs, curves, labels, title, file_name="errors.pdf", combined=True):
    # we assume all curves have the same length
    # if we use combined we also assume that loss is always the last
    if combined:
        fig, (axs, ax2) = plt.subplots(1, 2, sharex="all")
        ax2.grid(True)
    else:
        fig, axs = plt.subplots()

    x = np.arange(0, epochs)

    colors = ["mediumslateblue", "plum", "mediumslateblue"]
    for i in range(len(curves)):
        if i == len(curves) - 1 and combined:  # last elem
            ax2.plot(x, curves[i], label=labels[i], color=colors[i])

        else:
            axs.plot(x, curves[i], label=labels[i], color=colors[i])
            axs.legend()

    fig.suptitle(title)
    axs.grid(True)
    plt.xlim([0, epochs + 1])
    plt.subplots_adjust(wspace=0.4)
    plt.legend()
    plt.savefig("plots/" + file_name + ".svg")
    plt.show()


def accuracy(pos_preds, neg_preds):
    tresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    pos = np.zeros((2, len(tresholds)))  # [true positiveves,false negatives]
    neg = np.zeros((2, len(tresholds)))  # [true negatives,false positives]
    n = 0
    for treshold in tresholds:
        for res in pos_preds:
            if res > treshold:
                pos[0, n] += 1
            else:
                pos[1, n] += 1
        for res in neg_preds:
            if res > treshold:
                neg[1, n] += 1
            else:
                neg[0, n] += 1
        n += 1

    sens = pos[0] / (pos[1] + pos[0])
    spec = neg[0] / (neg[0] + neg[1])
    acc = (sens + spec) / 2
    fig, ax = plt.subplots()
    plt.plot(tresholds, acc, 'o-', color="mediumslateblue")
    print(acc)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Treshold for positive classification')
    ax.set_title('Accuracy of test set, proposed model')
    ax.grid(True)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig("plots/gnn_accuracy.svg")
    plt.show()


def plot_explain(relevances, src, tar, walks, pos, gamma):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))
    n = 0

    for node in nodes:
        graph.add_vertices(str(node))

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
        x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))

    place = np.array(list(graph.layout_kamada_kawai()))
    # edges plotting
    print(walks)
    print(place)
    fig, axs = plt.subplots()
    val_abs = 0
    max_abs = np.abs(max(map((lambda x: x.sum()), relevances)))

    sum_s = 0
    sum_t = 0
    sum_c = 0
    for walk in walks[:-1]:
        r = relevances[n]

        r = r.sum()
        if src in walk:
            sum_s += np.abs(r)
        if tar in walk:
            sum_t += np.abs(r)
        if tar in walk or src in walk:
            sum_c += np.abs(r)

        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0],
             place[nodes.index(walk[3]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1],
             place[nodes.index(walk[3]), 1]]
        tx, ty = utils.shrink(a, b)
        loops = utils_func.self_loops(a, b)
        loops.append((tx, ty))

        axs.arrow(a[0], b[0], a[1] - a[0], b[1] - b[0], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[1], b[1], a[2] - a[1], b[2] - b[1], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[2], b[2], a[3] - a[2], b[3] - b[2], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)

        for i in loops:
            if r > 0.0:
                alpha = np.clip((3 / max_abs) * r, 0, 1)
                axs.plot(i[0], i[1], alpha=alpha, color='indianred', lw=2.)

            if r < -0.0:
                alpha = np.clip(-(3 / max_abs) * r, 0, 1)
                axs.plot(i[0], i[1], alpha=alpha, color='slateblue', lw=2.)

        n += 1

        val_abs += np.abs(r)

    # nodes plotting
    for i in range(len(nodes)):
        axs.plot(place[i, 0], place[i, 1], 'o', color='black', ms=3)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='gold', ms=5, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], color='slateblue', label="negative relevance")
    axs.plot([], [], color='indianred', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")
    print(sum_s, sum_t, sum_c)
    gamma = str(gamma)
    gamma = gamma.replace('.', '')
    node = str(src)
    name = "LRP_plot_" + pos + "_example_" + node + gamma + "0.svg"
    plt.tight_layout()
    fig.savefig(name)
    fig.show()
    return val_abs


def validation(relevances: list, node):
    relevances = np.asarray(relevances)
    print(relevances)
    fig, axs = plt.subplots()
    axs.fill_between(np.arange(0, 25, 1), relevances[:, 1])
    axs.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25])

    plt.savefig("plots/validation_pru_" + str(node.numpy()))
    plt.show()


def tsne_plot():
    dataset = dataLoader.LinkPredData("data/", "big_graph", use_subset=False)
    data = dataset.load(transform=True)

    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    data_x = data.x[0:data.x.shape[0] // 2]
    print(data_x.shape, data.x.shape)
    embedding_train = tsne.fit(data_x)
    print(embedding_train, embedding_train.shape)

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(data.x)
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    fig, ax = plt.subplots()
    plt.tight_layout()
    plt.scatter(embedding_train[:, 0], embedding_train[:, 1], color="mediumslateblue")
    plt.axis("off")
    plt.savefig("plots/tsne.jpg")

    plt.show()

def plt_node_lrp(rel, src, tar,walks):
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))
    n = 0
    print(nodes)
    print(rel.shape)
    for node in nodes:
        graph.add_vertices(str(node))

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
        x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))

    place = np.array(list(graph.layout_kamada_kawai()))
    # edges plotting

    fig, axs = plt.subplots()
    val_abs = 0

    for walk in walks[:-1]:
        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0],
             place[nodes.index(walk[3]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1],
             place[nodes.index(walk[3]), 1]]
        tx, ty = utils.shrink(a, b)
        loops = utils_func.self_loops(a, b)
        loops.append((tx, ty))

        axs.arrow(a[0], b[0], a[1] - a[0], b[1] - b[0], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[1], b[1], a[2] - a[1], b[2] - b[1], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[2], b[2], a[3] - a[2], b[3] - b[2], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)

        n += 1

    # nodes plotting
    max_abs = max(np.abs(rel[nodes]))
    for i in range(len(nodes)):
        if rel[nodes[i]] > 0:
            alpha = np.clip((4 / max_abs) * rel[nodes[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='red', alpha=alpha, ms=3)
        else:
            alpha = np.clip(-(4 / max_abs) * rel[nodes[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='blue', alpha=alpha, ms=3)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='gold', ms=5, label="target node")

    plt.savefig("plots/lrp_node.jpeg")
    plt.show()

def reindex(nodes, edgeindex,src,tar):
    new = torch.arange(0,len(nodes))
    tmp = torch.vstack((torch.asarray(nodes),new)).T
    for i in range(tmp.shape[0]):
        tmp1 = np.flatnonzero(tmp[i][0] == edgeindex[0])
        tmp2 = np.flatnonzero(tmp[i][0] == edgeindex[1])
        if tmp[i][0] == tar : tar_new = tmp[i][1]
        if tmp[i][0] == src : src_new = tmp[i][1]

        # reindexing
        edgeindex[0, tmp1] = tmp[i][1]
        edgeindex[1, tmp2] = tmp[i][1]

    return new.tolist(), edgeindex, tar_new, src_new


def plt_gnnexp(rel, src, tar, walks):
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(rel)


    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))
    print(walks)
    for node in nodes:
        graph.add_vertices(str(node))

    #x, y = [], []
    #for i in range(edge_index.shape[1]):
    #    graph.add_edges([(edge_index[0, i], edge_index[1, i])])
    #    x.append(edge_index[0, i]), y.append(edge_index[1, i])

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
        x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))

    place = np.array(list(graph.layout_kamada_kawai()))
    fig, axs = plt.subplots()

    print(place)
    edge_weight = edge_weight.detach().numpy()
    max_abs = np.abs(max(edge_weight))
    for i in range(len(edge_weight)):
        a = [place[nodes.index(edge_index[0, i]), 0], place[nodes.index(edge_index[0, i]), 1]]
        b = [place[nodes.index(edge_index[1, i]), 0], place[nodes.index(edge_index[1, i]), 1]]
        if edge_weight[i] > 0.0:
            color = 'red'
        else:
            color = 'blue'
        alpha = np.clip((4 / max_abs) * np.abs(edge_weight[i]), 0, 1)

        axs.arrow(b[0], b[1], a[0] - b[0], a[1] - b[1], color=color, lw=0.5, alpha=alpha, length_includes_head=True,
                  head_width=0.075)

    for i in range(len(nodes)):
        axs.plot(place[i, 0], place[i, 1], 'o', color='grey', alpha=0.3, ms=3)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='gold', ms=5, label="target node")
    # nodes plotting
    # TODO Src & target

    plt.savefig("plots/gnn_exp_.jpeg")
    plt.show()


def plot_cam(rel, adj, src, tar,walks):
    rel = rel.detach().numpy()
    nodes = list(set(np.asarray(walks).flatten()))
    graph = igraph.Graph()

    for node in nodes:
        graph.add_vertices(str(node))


    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
        x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
        x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
        x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))

    place = np.array(list(graph.layout_kamada_kawai()))

    for walk in walks:
        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0],
             place[nodes.index(walk[3]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1],
             place[nodes.index(walk[3]), 1]]
        tx, ty = utils.shrink(a, b)
        loops = utils_func.self_loops(a, b)
        loops.append((tx, ty))

        axs.arrow(a[0], b[0], a[1] - a[0], b[1] - b[0], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[1], b[1], a[2] - a[1], b[2] - b[1], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        axs.arrow(a[2], b[2], a[3] - a[2], b[3] - b[2], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
    max_abs = max(np.abs(rel[nodes]))

    # TODO set edgecolor for src tar
    for i in range(len(nodes)):
        print(rel[nodes[i]])
        if rel[nodes[i]] > 0:
            alpha = np.clip((4 / max_abs) * rel[nodes[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='red', alpha=alpha, ms=3)
        else:
            alpha = np.clip(-(4 / max_abs) * rel[nodes[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='blue', alpha=alpha, ms=3)

    axs.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='gold', ms=5, label="target node")

    plt.savefig("plots/cam.jpeg")
    plt.show()

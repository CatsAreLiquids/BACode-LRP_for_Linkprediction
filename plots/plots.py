import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import igraph
import torch
import torch_geometric.utils
from openTSNE import TSNE
import scipy.sparse as ssp
from utils import utils_func, utils
from itertools import groupby
from typing import Dict, List
import torch_sparse
from matplotlib import patches, lines
import os
import glob

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
    #ax.set_yticks([0.0, 0.225, 0.45])
    ax.set_ylabel(r"$\sum f(x)$")
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig("plots/RelevanceAtDifLayers_without_5.svg")
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
    gamma = str(gamma)
    gamma = gamma.replace('.', '')
    node = str(src)
    name = "LRP_plot_" + pos + "_example_" + node +"withoutw_link"+ ".svg"
    plt.tight_layout()
    fig.savefig(name)
    fig.show()
    return val_abs

def side_by_side_plot(p, special_walks_indexes, src, tar, walks, gamma, structure):
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs : List[plt.Axes]
    structure = refactored_plot_explain(p, src, tar, walks, "pos", gamma, structure, use_structure= True, ax= axs[0])
    special_idx_p = [_p if j in special_walks_indexes else np.array(0) for j,_p in enumerate(p)]
    refactored_plot_explain(special_idx_p, src, tar, walks, "pos", gamma, structure, use_structure= True, ax= axs[1])
    axs[0].set_title(f"max: {max(p):.3f} - min: {min(p):.3f}")
    axs[1].set_title(f"max: {(max(special_idx_p)*100)/max(p):2.3f}% - min: {(min(special_idx_p)*100)/min(p):2.3f}%")
    axs[1].legend().remove()
    plt.show()
    return structure

def get_edge_index_of_subgraph(full_edge_index : np.ndarray, subgraph : np.ndarray) -> torch.Tensor:
    """
    Gets the edge index of a graph and a subgraph and returns the edge index of the subgraph
    """
    edge_index = full_edge_index.transpose()
    edge_index = edge_index[np.isin(edge_index[:, 0], subgraph) & np.isin(edge_index[:, 1], subgraph)]
    return torch.tensor(edge_index.transpose())    

def get_graph(adjacency_matrix : torch_sparse.SparseTensor, subgraph : np.ndarray, src : np.ndarray, tar : np.ndarray) -> igraph.Graph:
    """
    Gets the adjacency matrix of a graph and a subgraph and returns the adjacency matrix of the subgraph

    @param : adjacency_matrix : torch_sparse.SparseTensor, the adjacency matrix of the graph
    @param : subgraph : torch.Tensor, the nodes contanied in the subgraph
    """
    graph : igraph.Graph = igraph.Graph()
    vertex_names = []
    is_src = []
    is_tar = []
    for vertex in subgraph:
        vertex_names.append(str(vertex))
        is_src.append(True if vertex == src else False)
        is_tar.append(True if vertex == tar else False)
    graph.add_vertices(vertex_names, {'is_src' : is_src, 'is_tar' : is_tar})

    edge_index_transposed = get_edge_index_of_subgraph(np.stack(adjacency_matrix.coo()[:2]), subgraph).t().tolist()
    graph.add_edges(
        es= [[str(edge[1]), str(edge[0])] for edge in edge_index_transposed],
        attributes= {'src_to_tar' : [True if edge[1] == src.item() and edge[0] == tar.item()
                                     else False for edge in edge_index_transposed]},
        )
    return graph

def customize_graph(graph : igraph.Graph, src : str, tar : str) -> igraph.Graph:
    # Define the style of the vertices
    color = {src : "yellowgreen", tar : "gold"}
    graph.vs["color"] = [color.get(vertex, "gray") for vertex in graph.vs["name"]]
    size = {src : 0.15, tar : 0.15}
    graph.vs["size"] = [size.get(vertex, 0.1) for vertex in graph.vs["name"]]
    graph.vs["frame_width"] = 0

    # Define the style of the edges
    graph.es["width"] = [0.2 if src_to_tar else 0.1 for src_to_tar in graph.es["src_to_tar"]]
    graph.es["color"] = ["green" if src_to_tar else "gray" for src_to_tar in graph.es["src_to_tar"]]
   
    # Define miscelaneous style of graph plot
    visual_style = {}

    return graph, visual_style

def get_alpha(x, r_max, r_min = 0, new_max = 1, new_min = 0.1):
    alpha = new_min + ((new_max - new_min)/(r_max - r_min))*(x - r_min)
    if 100*x/r_max < 0.01: # if the relevance is smaller than 0.01% of the max relevance for the subgraph, set it to zero.
        alpha = 0
    return np.clip(alpha, new_min, new_max) # is necessary to avoid floating points errors that may go outside of the acceptable range

def plot_graph(graph, visual_style, walks, rel, ax, gamma, epsilon, noise = True, layout_dict : Dict[str, object] = None):
    if layout_dict is None:
        layout_dict = {
            'layout' : "kk",
        }
    max_abs = np.abs(rel).max()
    layout = graph.layout(**layout_dict)
    for walk, r in zip(walks[:-1], rel):
        if r == 0: continue
        points = get_walk_points(layout, graph.vs["name"], [str(node) for node in walk])
        alpha = get_alpha(np.abs(r), max_abs, new_min = 0.1)
        plot_walk_trace(points, ax, r, alpha, noise= noise)
    igraph.plot(graph, **visual_style, layout= layout, target= ax)
    src_tar_ind = np.array(graph.vs["is_src"]) ^ np.array(graph.vs["is_tar"])
    ax.scatter(np.array(layout)[src_tar_ind][:, 0], np.array(layout)[src_tar_ind][:, 1], color= np.array(graph.vs["color"])[src_tar_ind].tolist(), sizes= 10*np.array(graph.vs["size"])[src_tar_ind])

def plot_graph_sum(graph, visual_style, walks, rel, ax, gamma, epsilon):
    max_abs = np.abs(rel).max()
    igraph.plot(graph, **visual_style, target= ax)
    cummulative_walks = {}
    hashify = lambda x: tuple(tuple(y) for y in x) # Make list of lists hashable
    for walk, r in zip(walks[:-1], rel):
        points = get_walk_points(graph.layout("kk"), graph.vs["name"], [str(node) for node in walk])
        cummulative_walks[hashify(points)] = cummulative_walks.get(hashify(points), 0) + r
    for points, r in cummulative_walks.items():
        alpha = get_alpha(np.abs(r), max_abs, new_min = 0.1)
        plot_walk_trace(points, ax, r, alpha, noise= False)

def save_plots(
        adjacency_matrix : torch_sparse.SparseTensor,
        explanation : List[np.ndarray],
        src : torch.Tensor,
        tar : torch.Tensor,
        walks : List[List[int]],
        gamma : float = 0.0,
        epsilon : float = 0.0,
    ):
    subgraph = np.unique(np.asarray(walks).flatten())
    graph = get_graph(adjacency_matrix, subgraph, np.asarray(src), np.asarray(tar))
    fig, ax = plt.subplots(figsize=(12,12))
    graph, visual_style = customize_graph(graph, str(src.item()), str(tar.item()))
    plot_graph(graph, visual_style, walks, explanation, ax, gamma, epsilon)
    os.makedirs("all_plots/", exist_ok= True)
    fig.savefig(f"all_plots/{src.item()}_{tar.item()}_{str(gamma).replace('.', ',')}_{str(epsilon).replace('.', ',')}.pdf")

def simple_plot(
        adjacency_matrix : torch_sparse.SparseTensor,
        explanation : List[np.ndarray],
        src : torch.Tensor,
        tar : torch.Tensor,
        walks : List[List[int]],
        gamma : float = 0.0,
        epsilon : float = 0.0,
        ax : plt.Axes = None,
        set_legend : bool = False,
        legend_args : dict[str, object] = None,
    ) -> None:
    subgraph = np.unique(np.asarray(walks).flatten())
    graph = get_graph(adjacency_matrix, subgraph, np.asarray(src), np.asarray(tar))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        plot_flag = True
    else: plot_flag = False
    graph, visual_style = customize_graph(graph, str(src.item()), str(tar.item()))
    plot_graph(graph, visual_style, walks, explanation, ax, gamma, epsilon)
    if plot_flag:
        fig.tight_layout()
        fig.show()
    if set_legend:
        if legend_args is None:
            legend_args = {}
        set_ax_legend(ax, **legend_args)

def plot_from_filename(
        full_file_path : str,
        adjacency_matrix : torch_sparse.SparseTensor,
        ax : plt.Axes = None,
        set_legend : bool = False,
        legend_kwargs : dict[str,object] = None,
        ) -> None:
    """
    Takes a filename in form <path_to_file>/{src}_{tar}_{gamma}_{epsilon}.th
    plots the walks contained in the sparse tensor

    @param: full_file_path: str: The full file path.
    @param: adjacency_matrix: SparseTensor: The adjacency matrix in torch_sparse format.
    @param: ax: plt.Axes: The ax to plot in, if it's None a new figure will be created.
    @param: set_legend: bool: Flag to generate a legend on this subplot.
    @param: legend_kwargs: dict: The kwargs for the legend.
    """
    rel_matrix = torch.load(full_file_path)
    filename = os.path.splitext(full_file_path)[0].split('/')[-1]
    src, tar, gamma, epsilon = filename.split('_')
    gamma, epsilon = [float(num.replace(',', '.')) for num in [gamma, epsilon]]
    explanations = rel_matrix._values()
    src, tar = torch.tensor(int(src)), torch.tensor(int(tar))
    walks = rel_matrix._indices().T.tolist()
    simple_plot(adjacency_matrix= adjacency_matrix, explanation=np.array(explanations), src=src, tar=tar, walks=walks,
                 gamma= gamma, epsilon= epsilon, ax= ax, set_legend= set_legend, legend_args= legend_kwargs)


def plot_all_walks_in_folder(path_to_folder: str, adj_t: torch_sparse.SparseTensor, save: bool = False):
    already_plotted = []
    all_files = glob.glob(os.path.join(path_to_folder,f"*.th"))
    for file in all_files[1:]:
        filename = os.path.splitext(file)[0].split('/')[-1]
        src, tar, _, _ = filename.split('_')
        if f"{src}, {tar}" in already_plotted or src == 'all': continue
        else: already_plotted.append(f"{src}, {tar}")

        plot_all_parameters_for_src_tar(path_to_folder, adj_t, int(src), int(tar), loc='upper left', bbox_to_anchor=(-1.35, 1), prop={'size': 6}, save=f"all_plots/{src}_{tar}.pdf" if save else "")


def plot_all_parameters_for_src_tar(path_to_folder : str, adjacency_matrix : torch_sparse.SparseTensor, src : int, tar : int, save : str = "", **kwargs) -> None:
    files = glob.glob(os.path.join(path_to_folder,f"{src}_{tar}_*.th"))
    files = sorted(files, key= lambda s: [float(_s.replace(',', '.')) for _s in os.path.splitext(s)[0].split('_')[-2:]])
    fig, axs = plt.subplots(2, len(files)//2 + len(files)%2, figsize=(kwargs.get("figsize_multiplier", 1.5) * (len(files)//2 + len(files)%2), kwargs.get("figsize_multiplier", 1.5) * 1.8))
    axs = np.array(axs).flatten()
    ordered_params = []
    for i,(file, ax, letter) in enumerate(zip(files, axs, 'ABCDEFGHI')):
        ax : plt.Axes
        gamma, epsilon = os.path.splitext(file)[0].split('/')[-1].split('_')[-2:]
        gamma, epsilon = [float(param.replace(',', '.')) for param in [gamma, epsilon]]
        ordered_params.append(tuple([gamma, epsilon]))
        plot_from_filename(file, adjacency_matrix, ax)
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1], letter)
    if len(axs) > len(files):
        axs[-1].axis('off')
        correct_positioning_of_axes(axs)
        set_ax_legend(axs[len(axs)//2 + len(axs)%1], **kwargs)
    fig.suptitle([f"{b}: {a}" for a,b in zip(ordered_params, 'ABCDEFGHI')], fontsize= 7)
    if save != "":
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def correct_positioning_of_axes(axs : plt.Axes):
    half = len(axs) // 2 + len(axs) % 2
    axs_positions = [_ax.get_position() for _ax in axs]
    _delta = (axs_positions[0].get_points() + axs_positions[1].get_points())/2 - axs_positions[0].get_points()
    _delta = _delta[:, 0].mean() * 1.1
    for ax in axs[half:-1][::-1]:
        pos = ax.get_position().bounds
        new_bounds = list(pos)
        new_bounds[0] += _delta
        ax.set_position(new_bounds)


def plot_explanations_modular(
        adjacency_matrix : torch_sparse.SparseTensor,
        explanations : List[List[np.ndarray]],
        src : torch.Tensor,
        tar : torch.Tensor,
        walks : List[List[int]],
        gamma : float = 0.0,
        epsilon : float = 0.0,
        ):
    """
    Every tensor of relevances contained in explanations explains the same subgraph
    """  
    subgraph = np.unique(np.asarray(walks).flatten())
    graph = get_graph(adjacency_matrix, subgraph, np.asarray(src), np.asarray(tar))
    fig, axs = plt.subplots(len(explanations), 2, figsize=(10, 6*len(explanations)))
    axs = axs.reshape(2, -1).transpose()
    # fig.suptitle(f"Explanations for the walks from {src} to {tar} - gamma: {gamma} - epsilon: {epsilon}")
    for rel_explanation, ax in zip(explanations, axs):
        ax : List[plt.Axes]
        graph, visual_style = customize_graph(graph, str(src.item()), str(tar.item()))
        plot_graph(graph, visual_style, walks, rel_explanation, ax[0], gamma, epsilon)
        # ax[0].set_title(f"Explanation with slight shift")
        plot_graph_sum(graph, visual_style, walks, rel_explanation, ax[1], gamma, epsilon)
        # ax[1].set_title(f"Explanation with cummulative sum")
    axs[0][0].legend(
            handles= [
                lines.Line2D([], [], color='yellowgreen', linewidth= 0, marker= 'o', label= 'Source Node'),
                lines.Line2D([], [], color='gold', linewidth= 0, marker= 'o', label= 'Target Node'),
                lines.Line2D([], [], color='blue', linewidth= 1, label='Negative relevance'),
                lines.Line2D([], [], color='red', linewidth= 1, label='Positive relevance'),
                ],
            loc = 'upper left',
            bbox_to_anchor= (-0.3, 1.3)
            )

    return graph    

def set_ax_legend(ax : plt.Axes, loc= 'upper left', bbox_to_anchor= (-0.3, 1.3), **kwargs) -> None:
    ax.legend(
            handles= [
                lines.Line2D([], [], color='yellowgreen', linewidth= 0, marker= 'o', label= 'Source Node'),
                lines.Line2D([], [], color='gold', linewidth= 0, marker= 'o', label= 'Target Node'),
                lines.Line2D([], [], color='blue', linewidth= 1, label='Negative relevance'),
                lines.Line2D([], [], color='red', linewidth= 1, label='Positive relevance'),
                ],
            loc = loc,
            bbox_to_anchor= bbox_to_anchor,
            **kwargs
            )
    
def plot_explain_no_circles(
        adjacency_matrix : torch_sparse.SparseTensor,
        relevances : List[np.ndarray],
        src : torch.Tensor,
        tar : torch.Tensor,
        walks : List[List[int]],
        ax : plt.Axes,
        gamma : float = 0.0,
        epsilon : float = 0.0,
        set_legend : bool = False,
        **kwargs,
        ) -> None: 
    subgraph = np.unique(np.asarray(walks).flatten())
    graph = get_graph(adjacency_matrix, subgraph, np.asarray(src), np.asarray(tar))
    graph, visual_style = customize_graph(graph, str(src.item()), str(tar.item()))
    plot_graph(graph, visual_style, walks, relevances, ax, gamma, epsilon, noise= True)
    if set_legend:
        set_ax_legend(ax, **kwargs)

def refactored_plot_explain(relevances, src, tar, walks, pos, gamma, structure= None, use_structure= False, ax= None):
    if structure is None or use_structure is False:
        graph = igraph.Graph()
        nodes = list(set(np.asarray(walks).flatten()))
        for node in nodes:
            graph.add_vertices(str(node))
        x, y = [], []
        for walk in walks:
            graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])
            x.append(nodes.index(walk[0])), y.append(nodes.index(walk[1]))
            x.append(nodes.index(walk[1])), y.append(nodes.index(walk[2]))
            x.append(nodes.index(walk[2])), y.append(nodes.index(walk[1]))
        place = np.array(list(graph.layout_kamada_kawai()))
    else:
        graph = structure["graph"]
        nodes = structure["nodes"]
        x, y = structure["x_y"]
        place = structure["place"]

    n = 0 
    # edges plotting
    if ax is None:
        fig, ax = plt.subplots()
    # axs.set_xlim(-1.8, 1.8)
    # axs.set_ylim(-1.8, 1.8)
    val_abs = 0
    max_abs = np.abs(relevances).max()

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

        points = get_walk_points(place, nodes, walk)
        alpha = get_alpha(r, max_abs)
        plot_walk_trace(points, ax, r, alpha)
        a = [place[nodes.index(walk[0]), 0], place[nodes.index(walk[1]), 0], place[nodes.index(walk[2]), 0],
             place[nodes.index(walk[3]), 0]]
        b = [place[nodes.index(walk[0]), 1], place[nodes.index(walk[1]), 1], place[nodes.index(walk[2]), 1],
             place[nodes.index(walk[3]), 1]]
        tx, ty = utils.shrink(a, b)
        loops = utils_func.self_loops(a, b)
        # loops.append((tx, ty))

        ax.arrow(a[0], b[0], a[1] - a[0], b[1] - b[0], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        ax.arrow(a[1], b[1], a[2] - a[1], b[2] - b[1], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)
        ax.arrow(a[2], b[2], a[3] - a[2], b[3] - b[2], color='grey', lw=0.5, alpha=0.3, length_includes_head=True,
                  head_width=0.075)

        for i in loops:
            if r > 0.0:
                alpha = np.clip((3 / max_abs) * r, 0, 1)
                ax.plot(i[0], i[1], alpha=alpha, color='indianred', lw=2.)

            if r < -0.0:
                alpha = np.clip(-(3 / max_abs) * r, 0, 1)
                ax.plot(i[0], i[1], alpha=alpha, color='slateblue', lw=2.)

        n += 1

        val_abs += np.abs(r)
        # fig.savefig(f"animation/{n}.png")

    # nodes plotting
    for i in range(len(nodes)):
        ax.plot(place[i, 0], place[i, 1], 'o', color='black', ms=3)

    ax.plot(place[nodes.index(src), 0], place[nodes.index(src), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    ax.plot(place[nodes.index(tar), 0], place[nodes.index(tar), 1], 'o',
             color='gold', ms=5, label="target node")

    # legend shenenigans & # plot specifics
    ax.plot([], [], color='slateblue', label="negative relevance")
    ax.plot([], [], color='indianred', label="positive relevance")

    ax.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    ax.axis("off")
    # print(sum_s, sum_t, sum_c)
    gamma = str(gamma)
    gamma = gamma.replace('.', '')
    node = str(src)
    name = "LRP_plot_" + pos + "_example_" + node + gamma + "0.svg"
    # plt.tight_layout()
    # fig.savefig(name)
    # fig.show()
    return {'graph' : graph, 'nodes' : nodes, 'place' : place, 'x_y': [x,y]}

def get_walk_points(place, nodes, walk):
    trace_points = [key for key, _group in groupby(walk)]
    return [place[nodes.index(point)] for point in trace_points]

def plot_walk_trace(points : List[List[float]], ax : plt.Axes, rel : float, alpha : float, noise : bool = True) -> None:
    if len(points) < 2: return
    if len(points) == 2:
        codes = [
            Path.MOVETO,
            Path.LINETO,
        ]
    if len(points) == 3:
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
        ]
    if len(points) == 4:
        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]
    if noise:
        noise = np.random.normal(0, 0.033, size=(len(points), len(points[0])))
        points = np.asarray(points) + noise
    path = Path(points, codes)
    color = 'indianred' if np.sign(rel) > 0 else 'slateblue'
    patch = patches.PathPatch(path, edgecolor= color, alpha= 0 if np.isnan(alpha) else alpha, facecolor='none', lw=0.7)
    ax.add_patch(patch)

def validation(relevances: list, node):
    relevances = np.asarray(relevances)
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
    embedding_train = tsne.fit(data_x)

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


def plt_gnnexp(rel, src, tar, walks, mapping):
    edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(rel)
    graph = igraph.Graph()
    nodes = list(set(np.asarray(walks).flatten()))

    for node in nodes:
        graph.add_vertices(str(node))

    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])),
                         (str(walk[1]), str(walk[2])),
                         (str(walk[2]), str(walk[3]))])

    place = np.array(list(graph.layout_kamada_kawai()))
    fig, axs = plt.subplots()

    edge_weight = edge_weight.detach().numpy()
    max_abs = np.abs(max(edge_weight))

    for i in range(len(edge_weight)):
        a = [place[nodes.index(mapping[edge_index[0, i]]), 0], place[nodes.index(mapping[edge_index[0, i]]), 1]]
        b = [place[nodes.index(mapping[edge_index[1, i]]), 0], place[nodes.index(mapping[edge_index[1, i]]), 1]]
        if edge_weight[i] > 0.0:
            color = 'red'
        else:
            color = 'blue'
        alpha = np.clip((4 / max_abs) * np.abs(edge_weight[i]), 0, 1)

        axs.arrow(b[0], b[1], a[0] - b[0], a[1] - b[1], color=color, lw=0.5, alpha=alpha, length_includes_head=True,
                  head_width=0.075)

    for i in range(len(nodes)):
        axs.plot(place[i, 0], place[i, 1], 'o', color='grey', alpha=0.3, ms=3)

    axs.plot(place[nodes.index(mapping[src]), 0], place[nodes.index(mapping[src]), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(mapping[tar]), 0], place[nodes.index(mapping[tar]), 1], 'o',
             color='gold', ms=5, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], color='slateblue', label="negative relevance")
    axs.plot([], [], color='indianred', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    plt.savefig("plots/gnn_exp_.jpeg")
    plt.show()


def plot_cam(rel, src, tar,walks,mapping):
    rel = rel.detach().numpy()
    nodes = list(set(walks.flatten()))
    graph = igraph.Graph()
    for node in nodes:
        graph.add_vertices(str(node))

    fig, axs = plt.subplots()

    x, y = [], []
    for walk in walks:
        graph.add_edges([(str(walk[0]), str(walk[1])), (str(walk[1]), str(walk[2])), (str(walk[2]), str(walk[3]))])


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

    tmp = [mapping.index(x) for x in nodes]
    max_abs = max(np.abs(rel[tmp]))

    # TODO set edgecolor for src tar
    #TODO check if this is correct
    for i in range(len(tmp)):
        if rel[tmp[i]] > 0:
            alpha = np.clip((4 / max_abs) * rel[tmp[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='red', alpha=alpha, ms=3)
        else:
            alpha = np.clip(-(4 / max_abs) * rel[tmp[i]], 0, 1)
            axs.plot(place[i, 0], place[i, 1], 'o', color='blue', alpha=alpha, ms=3)

    axs.plot(place[nodes.index(mapping[src]), 0], place[nodes.index(mapping[src]), 1], 'o',
             color='yellowgreen', ms=5, label="source node")
    axs.plot(place[nodes.index(mapping[tar]), 0], place[nodes.index(mapping[tar]), 1], 'o',
             color='gold', ms=5, label="target node")

    # legend shenenigans & # plot specifics
    axs.plot([], [], color='slateblue', label="negative relevance")
    axs.plot([], [], color='indianred', label="positive relevance")

    axs.legend(loc=2, bbox_to_anchor=(-0.15, 1.14))
    axs.axis("off")

    plt.savefig("plots/cam.jpeg")
    plt.show()

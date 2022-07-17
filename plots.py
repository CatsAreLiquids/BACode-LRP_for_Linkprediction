import numpy as np
import matplotlib.pyplot as plt
import igraph
import utils
import numpy


def layers_sum(walks, gnn, r_src, r_tar, tar, x, edge_index, pred):
    arr = np.zeros((5, 1))
    arr[0] = pred.detach().sum()

    walks = np.asarray(walks)
    l = set(walks[:, 3])
    for node in l:
        res = gnn.lrp(x, edge_index, [node, node, node, node], r_src, r_tar, tar)
        print(res[0])
        arr[1] += res[0].numpy()
    l = set([tuple((walks[x, 2], walks[x, 3])) for x in range(walks.shape[0])])

    for node in l:
        res = gnn.lrp(x, edge_index, [node[0], node[0], node[0], node[1]], r_src, r_tar, tar)
        arr[2] += res[1].numpy()
    l = set([tuple((walks[x, 1], walks[x, 2], walks[x, 3])) for x in range(walks.shape[0])])

    for node in l:
        res = gnn.lrp(x, edge_index, [node[0], node[0], node[1], node[2]], r_src, r_tar, tar)
        arr[3] += res[2].numpy()

    for walk in walks:
        res = gnn.lrp(x, edge_index, walk, r_src, r_tar, tar)
        arr[4] += res[3].numpy()

    fig, ax = plt.subplots()
    ax.bar([0, 1, 2, 3, 4], arr.flatten().T, width=0.35, color="mediumslateblue")
    ax.set_xticks([0, 1, 2, 3, 4],
                  labels=["f(x)", r"$\sum R_J$", r"$\sum R_{JK}$", r"$\sum R_{JKL}$", r"$\sum R_{JKLM}$"])
    ax.set_yticks([0.0, 0.75, 1.50])
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

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Treshold for positive classification')
    ax.set_title('Accuracy of test set, proposed Model')
    ax.grid(True)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig("plots/gnn_accuracy.svg")
    plt.show()


def plot_explain(relevances, src, tar, walks, pos, gamma, data):
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
        loops = utils_func.selfl_oops(a, b)
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
    alpha_src = np.sqrt(((data[src].numpy() - data[nodes].numpy()) ** 2).sum(axis=1))
    alpha_src *= 1 / max(alpha_src)

    alpha_tar = np.sqrt(((data[tar].numpy() - data[nodes].numpy()) ** 2).sum(axis=1))
    alpha_tar *= 1 / max(alpha_tar)

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
    name = "plots/plots/LRP_plot_" + pos + "_example_" + node + gamma + "0.svg"
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


def sumlrp():
    lrp = [0., 2.15689068, 2.63837075, 2.46377611, 2.64981101,
           2.98370917, 2.92302637, 2.97478302, 2.95911373, 2.84913276,
           2.89955298, 2.91593897, 2.91172644, 2.89942345, 2.90979614,
           2.9320627, 2.91068534, 2.93396315, 2.91965755, 2.92493093,
           2.91713323, 2.9368174, 2.93392664, 2.95149651, 2.94216627,
           2.98931937, 2.99040575, 2.984083, 2.94827876, 2.94791751,
           2.96562514, 2.96410188, 2.9636951, 3.08509172, 3.08425063,
           3.05148094, 3.04953302, 3.12018184, 3.14245917, 3.2115551,
           3.19849671, 3.19762867, 3.19675929, 3.1969032, 3.19664214,
           3.19647183, 3.1965733, 3.19653679, 3.19641777, 3.19771836,
           3.19768781, 3.1977124, 3.19770274, 3.19769784, 3.1976988,
           3.19769667, 3.19815577]
    lrp0 = [0., 2.04263055, 2.72085368, 3.43196453, 3.81875582,
            3.84557503, 4.03542022, 4.17981213, 4.21994651, 4.23303662,
            4.21991469, 4.19164502, 4.21026365, 4.17030014, 4.16374773,
            4.15772104, 4.14535576, 4.1398702, 4.12954354, 4.12106143,
            4.1176373, 4.11279594, 4.34092639, 4.30601651, 4.19289874,
            4.13567049, 4.06283493, 4.05836895, 3.98349388, 3.86668634,
            3.7970081, 3.82058068, 3.7138484, 3.5988849, 3.60337101,
            3.54775201, 3.50573046, 3.41977643, 3.34642735, 3.32179529,
            3.31162408, 3.31131654, 3.31038633, 3.30541622, 3.3004154,
            3.29558006, 3.29760205, 3.23989612, 3.32972693, 3.30436125,
            3.28735951, 3.22375256, 3.22361221, 3.19956072, 3.19919591,
            3.1988592, 3.19815577]
    lrp0020 = [0., 2.22136009, 2.48638792, 3.19447013, 3.55514507,
               3.57541619, 3.69282128, 3.6886239, 3.83574739, 3.91440188,
               4.00626733, 4.02709677, 4.02200041, 3.99014741, 3.98707324,
               3.98000726, 3.98855341, 3.97929335, 3.98084114, 3.99684434,
               3.98839366, 3.97183337, 3.98876932, 4.02703293, 4.04417967,
               3.98761795, 3.89411264, 3.86646835, 3.84126461, 3.70577685,
               3.65461596, 3.67637653, 3.6277782, 3.49864374, 3.50190941,
               3.45871661, 3.44437091, 3.36337291, 3.3358847, 3.34819264,
               3.32019397, 3.3197063, 3.32450911, 3.3282551, 3.33099876,
               3.32969676, 3.32371827, 3.27545276, 3.2666825, 3.2250111,
               3.21989412, 3.21946942, 3.2015501, 3.19469774, 3.19460358,
               3.19376066, 3.19815577]
    lrp020 = [0., 0.81574347, 1.9931785, 2.42526784, 2.71348783,
              2.73464618, 2.86649147, 3.03910236, 2.98877503, 3.01235193,
              3.00555754, 3.04768109, 3.02454545, 3.01688631, 3.03143929,
              3.0734682, 3.2428893, 3.23846497, 3.22468165, 3.22488973,
              3.22372089, 3.21737273, 3.27010689, 3.24427501, 3.2077023,
              3.17807605, 3.01030434, 3.09371513, 3.07974427, 3.046408,
              3.09496732, 3.05925595, 3.10910015, 3.13857691, 3.14436695,
              3.07004555, 3.06355912, 3.10886407, 3.18312509, 3.18519701,
              3.17330694, 3.17231991, 3.17250412, 3.17055597, 3.16866832,
              3.17091973, 3.18080086, 3.12800923, 3.09669852, 3.10070271,
              3.14165669, 3.17893605, 3.16663125, 3.19664845, 3.19042481,
              3.18887028, 3.19815577]
    lrp1000 = [0., 2.5187205, 3.12527134, 3.12350558, 3.28362418,
               3.28854055, 3.21289341, 3.22244241, 3.29161844, 3.27633883,
               3.22259174, 3.20511391, 3.19574579, 3.19364313, 3.22730783,
               3.24361176, 3.22614093, 3.22824881, 3.24296515, 3.23614966,
               3.23678861, 3.23601427, 3.23151375, 3.22868216, 3.23411047,
               3.23010257, 3.23035187, 3.22106523, 3.15640856, 3.19749749,
               3.21690128, 3.23190575, 3.22790138, 3.22800338, 3.23143796,
               3.23038935, 3.22877785, 3.18886035, 3.18705336, 3.20894612,
               3.20592522, 3.20543829, 3.2054213, 3.205493, 3.20563217,
               3.2046519, 3.20503002, 3.20489251, 3.20405933, 3.20463137,
               3.20379083, 3.20416784, 3.20439343, 3.2039995, 3.20400923,
               3.2040029, 3.19815577]
    lrp10002 = [0., -0.16988415, 0.42188665, 0.64336798, 0.70092138,
                0.98015693, 1.16778246, 1.29436309, 1.28803543, 1.24366887,
                1.23743058, 1.22540721, 1.20601114, 1.21471563, 1.233012,
                1.25227125, 1.22675355, 1.23777908, 1.26402016, 1.26317253,
                1.30830746, 1.31604613, 1.32571554, 1.56211159, 1.68855592,
                1.76795138, 1.77573605, 1.76784232, 1.84507408, 2.32401885,
                2.28842732, 2.27209634, 2.29095857, 2.41941971, 2.51951476,
                2.74358204, 2.6028686, 2.75599777, 2.77494168, 2.77453935,
                2.83267663, 2.83239622, 2.83245019, 2.83291463, 2.83319399,
                2.83303667, 2.83313335, 2.8334652, 2.85132293, 2.9036849,
                2.90401778, 2.90318211, 2.90257822, 3.006159, 3.006159,
                3.006159, 3.19815577]
    lrp0202 = [0., -0.16988415, 0.42188665, 0.64336798, 0.70092138,
               0.98015693, 1.16778246, 1.29436309, 1.28803543, 1.24366887,
               1.23743058, 1.22540721, 1.20601114, 1.21471563, 1.233012,
               1.25227125, 1.22675355, 1.23777908, 1.26402016, 1.26317253,
               1.30830746, 1.31604613, 1.32571554, 1.56211159, 1.68855592,
               1.76795138, 1.77573605, 1.76784232, 1.84507408, 2.32401885,
               2.28842732, 2.27209634, 2.29095857, 2.41941971, 2.51951476,
               2.74358204, 2.6028686, 2.75599777, 2.77494168, 2.77453935,
               2.83267663, 2.83239622, 2.83245019, 2.83291463, 2.83319399,
               2.83303667, 2.83313335, 2.8334652, 2.85132293, 2.9036849,
               2.90401778, 2.90318211, 2.90257822, 3.006159, 3.006159,
               3.006159, 3.19815577]
    lrp00202 = [0., 0.89846913, 1.78287147, 1.84635201, 2.02036703,
                1.85878095, 1.84990813, 1.83540472, 1.90454731, 1.82164255,
                1.83025462, 1.86541621, 1.85300244, 1.84907968, 1.84892092,
                1.87122796, 1.94798438, 1.96002898, 1.95612406, 1.90096674,
                1.90795542, 1.92225182, 1.93891479, 1.9237393, 2.27585631,
                2.36397856, 2.30791693, 2.31053825, 2.37663212, 2.72319326,
                2.84943256, 2.80355815, 2.81707803, 2.78088036, 2.97247762,
                2.97174612, 2.98422598, 3.01113204, 3.01676285, 3.07911046,
                3.11638754, 3.1163629, 3.11740608, 3.11717276, 3.1170685,
                3.11714782, 3.11712301, 3.08848179, 3.10632202, 3.13362392,
                3.14904264, 3.19147798, 3.19565127, 3.19743991, 3.19743991,
                3.19763903, 3.19815577]
    lrp002 = [0., 1.16293485, 1.37206395, 1.85746455, 1.87787791,
              1.80201401, 1.92452157, 1.87146635, 1.80208395, 1.86923941,
              1.73459664, 1.7524352, 1.75114292, 1.76437685, 1.76752681,
              1.75639069, 1.84096359, 1.84465694, 1.94147523, 1.96598931,
              1.98636589, 1.99099223, 2.00545412, 2.01022667, 2.1202296,
              2.21249834, 2.16470642, 2.1674063, 2.17447917, 2.53495681,
              2.65930488, 2.66080149, 2.59850891, 2.6107885, 2.73518688,
              2.95531713, 2.98184642, 3.08034417, 3.07283136, 3.15323904,
              3.14980331, 3.14979345, 3.1489567, 3.14966172, 3.14960318,
              3.14935923, 3.14952347, 3.14884418, 3.14951241, 3.14884904,
              3.1489179, 3.19147798, 3.19565127, 3.19743991, 3.19743991,
              3.19763903, 3.19815577]
    lrp050 = [0., 0.88961263, 1.86959847, 2.40038788, 2.83533677,
              2.89079203, 3.0635567, 3.0642062, 3.11661697, 3.13433115,
              3.1144742, 3.15437219, 3.14118424, 3.10546929, 3.11003851,
              3.10335601, 3.115334, 3.07784974, 3.07710013, 3.28866524,
              3.50059734, 3.42881941, 3.30978231, 3.34417192, 3.39454496,
              3.40798347, 3.37196347, 3.31395977, 3.22909945, 3.17836933,
              3.24505091, 3.18081767, 3.12168279, 3.03790275, 3.06238197,
              3.07124787, 3.0285506, 3.00343323, 3.02689148, 3.02851365,
              3.02869459, 3.02832569, 3.03044087, 3.03024643, 3.02892825,
              3.03156623, 2.99095935, 3.02652844, 3.0265175, 3.0120325,
              3.04862665, 3.10209889, 3.1256646, 3.17273262, 3.18489038,
              3.19376066, 3.19815577]
    arr0 = np.asarray(
        [np.asarray(lrp0).sum(), np.asarray(lrp0020).sum(), np.asarray(lrp020).sum(), np.asarray(lrp050).sum(),
         np.asarray(lrp1000).sum()])
    arr02 = np.asarray(
        [np.asarray(lrp002).sum(), np.asarray(lrp00202).sum(), np.asarray(lrp0202).sum(), np.asarray(lrp).sum(),
         np.asarray(lrp10002).sum()])
    labels = [r"$\gamma = 0$", r"$\gamma = 0.02$", r"$\gamma = 0.2$", r"$\gamma = 0.5$", r"$\gamma = 100$"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    width = 0.35
    ax.bar(x - width / 2, arr0, 0.35, label=r"$\epsilon=0.0$", color="mediumslateblue")
    ax.bar(x + width / 2, arr02, 0.35, label=r"$\epsilon=0.2$", color="plum")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r"$\sum f(x)$")
    ax.set_xticks(x, labels)
    ax.legend()
    """
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    ax.bar([0, 1, 2, 3,4], arr.flatten().T, 0.35, color="mediumslateblue")
    ax.set_xticks([0, 1, 2, 3,4], labels=["(0,0)", "(0.02,0)", "(0.2,0)","(0.05,0)","(100,0)" ])
    """
    plt.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig("plots/lrp_sum_0_act.svg")
    plt.show()

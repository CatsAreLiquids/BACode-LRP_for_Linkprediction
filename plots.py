import numpy as np
import matplotlib.pyplot as plt
import igraph
import utils
import numpy
import utils_func


def layers_sum(walks, gnn, r_src, r_tar, tar, x, edge_index, pred):
    arr = np.zeros((5, 1))
    arr[0] = pred.detach().sum()
    walks = np.asarray(walks)
    l = set(walks[:, 3])

    for node in l:
        res = gnn.lrp(x, edge_index, [node, node, node, node], r_src, r_tar, tar)
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

        r = r.sum().numpy()
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
    lrp = [0.00000000e+00, 2.54313151e-07, 3.17891439e-07, 1.69038773e-05,
       1.74383322e-05, 3.03208828e-05, 7.89266825e-04, 1.09686852e-03,
       1.36878590e-03, 1.41073962e-03, 2.64486571e-03, 3.65262131e-03,
       3.25310528e-03, 3.37559779e-03, 6.39759302e-03, 9.69792604e-03,
       1.06051485e-02, 1.65387998e-02, 1.92627013e-02, 2.61179537e-02,
       3.44137112e-02, 2.69742211e-02, 4.73969966e-02, 8.92090499e-02,
       1.56445855e-01, 2.89965761e-01, 3.50680678e-01, 4.64973892e-01,
       7.72696234e-01, 1.00628692e+00, 1.05473750e+00, 1.25798100e+00,
       1.44691719e+00, 1.45098383e+00, 1.65639855e+00, 1.67426365e+00,
       1.72531819e+00, 2.02307315e+00, 2.12010658e+00, 2.09434734e+00,
       2.19906068e+00, 2.20083130e+00, 2.20158701e+00, 2.20364223e+00,
       2.20596901e+00, 2.20324908e+00, 2.21121763e+00, 2.23179778e+00,
       2.26130565e+00, 2.27991701e+00, 2.34243671e+00, 2.54038264e+00,
       2.55131961e+00, 2.67959940e+00, 2.67959940e+00, 2.71305674e+00,
       3.02231949e+00]
    lrp0 = [0.00000000e+00, 0.00000000e+00, 2.28484472e-08, 8.81155332e-07,
       2.17556953e-06, 2.88287799e-06, 1.33055647e-03, 2.33261685e-03,
       2.68577635e-03, 4.35544352e-03, 6.46133920e-03, 6.83547755e-03,
       8.24209948e-03, 8.90711546e-03, 1.17853890e-02, 1.10560745e-02,
       1.04054312e-02, 1.18954599e-02, 2.71348059e-02, 3.42950622e-02,
       3.57212861e-02, 3.40758214e-02, 5.28051674e-02, 8.66717041e-02,
       2.72929835e-01, 2.64936825e-01, 3.41687304e-01, 4.09377965e-01,
       7.70606682e-01, 1.09860576e+00, 1.16642028e+00, 1.31890302e+00,
       1.50487548e+00, 1.50590762e+00, 1.57461595e+00, 1.71844984e+00,
       1.68835055e+00, 1.93602226e+00, 2.12202092e+00, 2.15725629e+00,
       2.19951431e+00, 2.20153196e+00, 2.20486157e+00, 2.22360218e+00,
       2.24331071e+00, 2.28478607e+00, 2.26521442e+00, 2.26504130e+00,
       2.26120529e+00, 2.27826127e+00, 2.33760879e+00, 2.57732238e+00,
       2.55131961e+00, 2.67959940e+00, 2.67959940e+00, 2.71305674e+00,
       3.02231949e+00]
    lrp0020 = [0.00000000e+00, 1.66098277e-06, 1.15275383e-05, 1.15593274e-05,
       1.15831693e-05, 2.81373660e-05, 2.90358067e-04, 3.42444579e-04,
       6.41874472e-04, 7.20372796e-04, 1.10149384e-03, 1.45876805e-03,
       3.24105124e-03, 5.28334379e-03, 5.06280164e-03, 5.90034624e-03,
       5.71207404e-03, 7.18891422e-03, 8.87165566e-03, 1.95636084e-02,
       2.53560503e-02, 2.91229089e-02, 4.66845602e-02, 8.57713759e-02,
       1.41926640e-01, 2.99140905e-01, 3.42331955e-01, 5.26791510e-01,
       7.72471214e-01, 1.04170870e+00, 1.13159739e+00, 1.25559815e+00,
       1.46605361e+00, 1.48560359e+00, 1.57980615e+00, 1.69259920e+00,
       1.68728703e+00, 1.93502907e+00, 2.12140170e+00, 2.15640366e+00,
       2.19912067e+00, 2.20030008e+00, 2.20424797e+00, 2.20428283e+00,
       2.21025795e+00, 2.24360562e+00, 2.27898036e+00, 2.26617816e+00,
       2.27009724e+00, 2.28849077e+00, 2.33815159e+00, 2.58260981e+00,
       2.55131961e+00, 2.67959940e+00, 2.67959940e+00, 2.71305674e+00,
       3.02231949e+00]
    lrp020 = [0.00000000e+00, 7.94728597e-09, 1.69575214e-05, 1.74661477e-05,
               1.75893307e-05, 1.78794066e-05, 1.85231368e-05, 2.43804852e-04,
               3.48232190e-04, 4.02221084e-04, 5.88217378e-04, 8.07562470e-04,
               2.66496638e-03, 3.98999254e-03, 5.36755125e-03, 9.87910032e-03,
               9.20565526e-03, 1.30861481e-02, 1.96605921e-02, 1.43587987e-02,
               1.81397796e-02, 3.05853287e-02, 9.36439703e-02, 1.86341483e-01,
               3.00981305e-01, 3.08432075e-01, 3.55406302e-01, 4.46333401e-01,
               7.34741430e-01, 1.11070160e+00, 1.17578542e+00, 1.33192750e+00,
               1.50696021e+00, 1.51780047e+00, 1.68379651e+00, 1.82726607e+00,
               1.87820880e+00, 2.02588559e+00, 2.19247648e+00, 2.15916345e+00,
               2.20187135e+00, 2.20240492e+00, 2.20369655e+00, 2.23412278e+00,
               2.23543162e+00, 2.21765944e+00, 2.21638582e+00, 2.27892446e+00,
               2.26310838e+00, 2.31395338e+00, 2.39933969e+00, 2.65927286e+00,
               2.66856966e+00, 2.67959940e+00, 2.67959940e+00, 2.71305674e+00,
               3.02231949e+00]
    lrp050 = [0.00000000e+00, 2.30471293e-07, 1.87873840e-05, 3.09373935e-04,
       3.09529901e-04, 1.05051200e-03, 1.06502374e-03, 1.08915567e-03,
       1.09175742e-03, 1.38707062e-03, 1.58925454e-03, 2.72393425e-03,
       6.51986301e-03, 1.78816527e-02, 1.88395590e-02, 2.20683267e-02,
       2.22787788e-02, 2.69290576e-02, 2.72727142e-02, 5.25285800e-02,
       5.92085083e-02, 7.34095057e-02, 1.09531840e-01, 1.61723669e-01,
       2.46083519e-01, 3.04700073e-01, 3.43876303e-01, 4.57027204e-01,
       7.05594966e-01, 1.10547387e+00, 1.27920581e+00, 1.43066092e+00,
       1.61610191e+00, 1.66404761e+00, 1.69849502e+00, 1.75867280e+00,
       1.81117643e+00, 2.07157878e+00, 2.09251987e+00, 2.15581624e+00,
       2.20166990e+00, 2.20189316e+00, 2.20376066e+00, 2.22427268e+00,
       2.27715371e+00, 2.27192714e+00, 2.27472855e+00, 2.32447369e+00,
       2.33024797e+00, 2.37618530e+00, 2.38982287e+00, 2.62542804e+00,
       2.64580661e+00, 2.67905660e+00, 2.67959940e+00, 2.67959940e+00,
       3.02231949e+00]
    lrp1000 = [0.00000000e+00, 0.00000000e+00, 3.67561976e-08, 1.43051147e-07,
       4.29153442e-07, 6.31229083e-04, 1.09885136e-03, 2.76842117e-03,
       2.66805689e-03, 5.58158358e-03, 4.97904817e-03, 5.04845877e-03,
       5.22908469e-03, 6.32935663e-03, 8.67204964e-03, 9.66835618e-03,
       1.10755940e-02, 1.12739702e-02, 1.87900364e-02, 2.45556265e-02,
       4.61612791e-02, 4.52951243e-02, 5.48336873e-02, 9.97683793e-02,
       2.86341163e-01, 3.52471040e-01, 3.83169412e-01, 4.64670870e-01,
       6.16571930e-01, 1.01313755e+00, 1.02845626e+00, 1.25441414e+00,
       1.41300058e+00, 1.51472318e+00, 1.60489781e+00, 1.68041497e+00,
       1.75247704e+00, 2.03579216e+00, 2.14032878e+00, 2.15673942e+00,
       2.19985967e+00, 2.20030348e+00, 2.20165244e+00, 2.20452682e+00,
       2.20484921e+00, 2.21013460e+00, 2.20723787e+00, 2.23050066e+00,
       2.26273803e+00, 2.28046003e+00, 2.37950456e+00, 2.55584922e+00,
       2.55077681e+00, 2.67959940e+00, 2.67959940e+00, 2.67959940e+00,
       3.02231949e+00]
    lrp10002 = [0.00000000e+00, 5.24520874e-07, 6.03000323e-07, 7.65919685e-07,
       9.96589661e-06, 2.66373158e-05, 3.42230002e-05, 3.75052293e-05,
       7.99204906e-04, 7.29578733e-04, 1.08573536e-03, 1.51091119e-03,
       2.74583399e-03, 3.17901671e-03, 4.15464938e-03, 7.09539453e-03,
       7.12743004e-03, 1.06968900e-02, 2.18603830e-02, 2.72478322e-02,
       3.03579966e-02, 4.01662538e-02, 4.33206012e-02, 1.49079417e-01,
       2.65672204e-01, 2.90857189e-01, 3.25513957e-01, 4.00854032e-01,
       6.50838261e-01, 1.03142506e+00, 1.12241749e+00, 1.33250224e+00,
       1.50873205e+00, 1.53168317e+00, 1.61568871e+00, 1.68506472e+00,
       1.73832425e+00, 2.03379026e+00, 2.15130903e+00, 2.15717754e+00,
       2.19941003e+00, 2.20024691e+00, 2.20273977e+00, 2.20619984e+00,
       2.20646899e+00, 2.20740992e+00, 2.20563319e+00, 2.22963456e+00,
       2.26164804e+00, 2.27937004e+00, 2.37865910e+00, 2.53983984e+00,
       2.55131961e+00, 2.67959940e+00, 2.67959940e+00, 2.67959940e+00,
       3.02231949e+00]
    lrp0202 = [0.00000000e+00, 3.97364299e-09, 8.54333242e-08, 8.54333242e-08,
       6.37769699e-07, 1.52615209e-03, 1.80928210e-03, 2.25089689e-03,
       2.32810875e-03, 2.35117972e-03, 3.42996816e-03, 4.99327083e-03,
       4.95814880e-03, 4.97955581e-03, 5.85587919e-03, 6.15556339e-03,
       1.29907151e-02, 1.37066712e-02, 1.54441575e-02, 2.22239504e-02,
       2.52470811e-02, 2.23101377e-02, 4.64402546e-02, 9.60091730e-02,
       1.44561597e-01, 2.82467010e-01, 3.36891060e-01, 3.76298691e-01,
       6.70714105e-01, 1.00473712e+00, 1.02685252e+00, 1.15862061e+00,
       1.45851584e+00, 1.46327883e+00, 1.55159862e+00, 1.73825010e+00,
       1.75148815e+00, 2.03396264e+00, 2.12055475e+00, 2.15578272e+00,
       2.19985056e+00, 2.20382531e+00, 2.20572796e+00, 2.20694651e+00,
       2.20695352e+00, 2.20845494e+00, 2.23645565e+00, 2.22870331e+00,
       2.24525985e+00, 2.27906786e+00, 2.30107146e+00, 2.53983984e+00,
       2.55077681e+00, 2.67959940e+00, 2.67959940e+00, 2.67959940e+00,
       3.02231949e+00]
    lrp00202 = [0.00000000e+00, 5.96046448e-09, 1.37446324e-04, 1.37525797e-04,
       1.38626496e-04, 1.56428417e-04, 1.60545111e-04, 1.70201063e-04,
       3.14503014e-03, 3.20780575e-03, 3.48930260e-03, 2.19641427e-03,
       1.66676243e-03, 4.84200517e-03, 5.07436097e-03, 6.37671451e-03,
       2.38405138e-02, 2.99719999e-02, 2.92747080e-02, 2.93257167e-02,
       2.43927797e-02, 2.73009618e-02, 4.41662620e-02, 4.58653271e-02,
       1.42871250e-01, 2.71689396e-01, 3.64779383e-01, 5.32067949e-01,
       7.44482184e-01, 1.03799215e+00, 1.08591648e+00, 1.18293493e+00,
       1.46977960e+00, 1.46641408e+00, 1.54455194e+00, 1.61767204e+00,
       1.72599032e+00, 1.93326961e+00, 2.14044703e+00, 2.15661132e+00,
       2.20011073e+00, 2.19961245e+00, 2.20155725e+00, 2.20529001e+00,
       2.20695352e+00, 2.20832856e+00, 2.21166949e+00, 2.22845184e+00,
       2.24552919e+00, 2.28053556e+00, 2.33856356e+00, 2.54798875e+00,
       2.54859526e+00, 2.67959940e+00, 2.67959940e+00, 2.71305674e+00,
       3.02231949e+00]
    lrp002 = [0.00000000e+00, 3.97364299e-09, 1.37476126e-04, 1.37507915e-04,
       1.37983759e-04, 1.38511260e-04, 1.40199065e-04, 2.41308908e-03,
       2.86571880e-03, 3.07348271e-03, 3.84034812e-03, 2.63139109e-03,
       1.80630783e-03, 4.86215850e-03, 6.28591577e-03, 8.35560759e-03,
       9.88473197e-03, 9.69660878e-03, 1.84303502e-02, 2.07897832e-02,
       3.41558019e-02, 3.62128615e-02, 4.96789893e-02, 7.74201016e-02,
       1.40627826e-01, 2.80074844e-01, 3.23432225e-01, 5.47556744e-01,
       7.76566900e-01, 1.01171210e+00, 1.03396487e+00, 1.16497306e+00,
       1.48486761e+00, 1.47978588e+00, 1.55965667e+00, 1.62709211e+00,
       1.77699731e+00, 1.97055351e+00, 2.13998396e+00, 2.15666414e+00,
       2.19964347e+00, 2.19901227e+00, 2.20082412e+00, 2.20385426e+00,
       2.20574775e+00, 2.20639019e+00, 2.23364545e+00, 2.26504425e+00,
       2.26000974e+00, 2.28085427e+00, 2.33992303e+00, 2.54858771e+00,
       2.55830153e+00, 2.68639563e+00, 2.68801863e+00, 2.71305674e+00,
       3.02231949e+00]

    random = [0.00000000e+00, 3.17891439e-07, 2.87729104e-03, 3.02552780e-03,
              3.19939852e-03, 5.52850962e-03, 1.83382054e-02, 1.93553547e-02,
              1.95349077e-02, 1.96838280e-02, 2.30122487e-02, 2.29943474e-02,
              3.43068769e-02, 5.16654025e-02, 5.24370939e-02, 5.04693220e-02,
              1.66488833e-01, 1.68820754e-01, 1.70465711e-01, 1.83869317e-01,
              1.97729564e-01, 2.43202978e-01, 2.60927276e-01, 3.10736990e-01,
              4.83839941e-01, 4.62710776e-01, 5.22312615e-01, 5.23060256e-01,
              8.57866615e-01, 1.24836267e+00, 1.51795260e+00, 1.60991675e+00,
              1.62556786e+00, 1.70989207e+00, 1.90854094e+00, 2.01148159e+00,
              2.08716923e+00, 2.16068107e+00, 2.16122442e+00, 2.20707686e+00,
              2.25016650e+00, 2.25013613e+00, 2.24608182e+00, 2.26458458e+00,
              2.28326272e+00, 2.28305771e+00, 2.29048947e+00, 2.38350404e+00,
              2.36266457e+00, 2.43699016e+00, 2.43605949e+00, 2.55495957e+00,
              2.73239944e+00, 2.87238565e+00, 2.96974197e+00, 2.96974031e+00,
              3.02231949e+00]
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

    plt.savefig("plots/lrp_sum_0_pru.pdf")
    plt.show()

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn
from ogb.linkproppred import Evaluator
from torch.nn.functional import relu

import dataLoader
import utils_func
import utils
import plots


class Baseline(torch.nn.Module):

    def __init__(self):
        super(Baseline, self).__init__()
        self.input = torch.nn.Linear(768, 256, bias=True)
        self.hidden = torch.nn.Linear(256, 256, bias=True)
        self.output = torch.nn.Linear(256, 1, bias=True)

    def forward(self, x):
        h = self.input(x)

        X = relu(h)
        h = self.hidden(X)

        X = relu(h)
        h = self.output(X)

        return h

    def lrp(self, rep, relevance, epsilon=0.0, gamma=0.0):
        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                print(cp.weight.sum())
                cp.weight[:, :] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        R[-1] = relevance
        print("test")
        rep = rep.data.clone().requires_grad_(True)

        A[0] = relu(rep)
        A[0] = A[0].data.clone().requires_grad_(True)
        A[1] = relu(self.input(rep)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1])).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2])
        print("z", z.sum())
        print("R2", R[2].sum())
        print("R2/z", R[2] / z.sum())
        s = R[2] / (z + 1e-15)
        print("s", s.sum())
        (z * s.data).sum().backward()
        print("a2", A[2].sum())
        print("a2grad", A[2].grad.sum())
        print("R1 manual", (A[2] * A[2].grad).sum())
        c = A[2].grad
        R[1] = A[2] * c
        print("R", R[1].sum())

        z = epsilon + roh(self.hidden).forward(A[1])
        print(z.shape)
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        z = epsilon + roh(self.input).forward(A[0])
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c

        """
        
        z = epsilon + roh(self.input).forward(relu(src + tar))
        print(z.shape)
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        src_grad = src.grad
        tar_grad = tar.grad
        """
        print(R[0].sum(), R[1].sum(), R[2].sum())

        return test


def explains(test_set, mlp, x, adj, rep):
    src, tar = test_set["source_node"], test_set["target_node_neg"][:, 0]
    walks_all = utils.walks(adj.t().to_dense())
    walks_all += utils.walks(adj.to_dense())
    # forward passes
    preds = mlp(helper(rep, x, src, tar, train=False))
    # print(preds)
    # print(preds)
    samples = [8, 107, 14, 453]
    # print(preds[samples])
    # samples=[preds.shape[0]-7]
    mean_r = 0
    for i in samples:
        # walks = utils_func.find_walks(src[i], tar[i], walks_all)
        print("?", preds[i])
        R = mlp.lrp(helper(rep, x, src[i], tar[i]), preds[i])
        mean_r += R
        utils_func.NN_res(R, i)
        """
        oneHopSrc, twoHopSrc = utils_func.get_nodes(adj,src[i])
        oneHopTar, twoHopTar = utils_func.get_nodes(adj, tar[i])
        utils_func.plot_explain_nodes(walks, src[i], tar[i], oneHopSrc, twoHopSrc, oneHopTar, twoHopTar,R)
        """
    utils_func.NN_res(mean_r / len(samples), "mean")


def run_cn(evaluator, data_set, adj):
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]
    pos_preds, neg_preds = [], []

    def cn(src, tar, adj):
        tmp_src = torch.reshape(torch.cat((adj[src], adj[:, src].T)), (1, adj.shape[0] * 2))
        tmp_tar = torch.reshape(torch.cat((adj[tar], adj[:, tar].T)), (1, adj.shape[0] * 2))

        if torch.count_nonzero(tmp_src) < 1 or torch.count_nonzero(tmp_tar) < 1:
            return np.array(0)

        tmp_src = torch.nonzero(tmp_src)[:, 1].numpy()
        tmp_tar = torch.nonzero(tmp_tar)[:, 1].numpy()

        res = np.intersect1d(tmp_tar, tmp_src).tolist()
        if len(res) < 1:
            return np.array(0)
        return np.array(len(res))

    for i in range(src.shape[0]):  # we dont need permutations
        src_tmp = src[i]
        tar_tmp = tar[i]
        tar_neg_tmp = tar_neg[i]

        # positive sampling
        pos_preds += [cn(src_tmp, tar_tmp, adj).squeeze()]
        # negative sampling
        for j in range(tar_neg_tmp.shape[0]):
            neg_preds += [cn(src_tmp, tar_neg_tmp[j], adj).squeeze()]
    pos_pred = np.asarray(pos_preds)
    neg_pred = np.asarray(neg_preds).reshape((-1, 20))

    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def helper(rep, data, src, tar, train=True):
    if src.ndim >= 1:
        dim = src.shape[0]
    else:
        dim = 1
    if train:
        x = torch.cat(
            (rep[src, 128:256], rep[src, 0:128] - data[tar], data[src], data[tar], rep[tar, 0:128] - data[src],
             rep[tar, 128:256]))
    else:
        x = torch.cat((
            rep[src, 128:256], rep[src, 0:128], data[src], data[tar], rep[tar, 0:128], rep[tar, 128:256]))

    return torch.reshape(x, (dim, 128 * 6)).to(torch.float32)


def train(optimizer, train_set, rep, data, mlp, batchsize):
    optimizer.zero_grad()
    train_src, train_tar = train_set["source_node"], train_set["target_node"]
    total_loss = []
    num_sample = 0
    permutation = torch.randperm(train_src.shape[0])

    for i in range(0, train_src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src = train_src[idx]
        tar = train_tar[idx]
        tar_neg = torch.randint(low=0, high=22064, size=tar.size(), dtype=torch.int64)

        # forward passes
        # positive sampling
        out = torch.sigmoid(mlp(helper(rep, data, src, tar, train=True)))
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        # negative sampling
        out = torch.sigmoid(mlp(helper(rep, data, src, tar_neg, train=True)))
        neg_loss = torch.log(1 - out + 1e-15)
        neg_loss = - torch.mean(neg_loss)

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize

    return sum(total_loss) / num_sample


@torch.no_grad()
def test(evaluator, data_set, mlp, rep, data, batchsize, accuracy=True):
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]
    permutation = torch.randperm(src.shape[0])

    pos_preds, neg_preds = [], []

    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [mlp(helper(rep, data, src_tmp, tar_tmp, train=False)).squeeze().cpu()]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [mlp(helper(rep, data, src_tmp, tar_neg_tmp, train=False)).squeeze().cpu()]

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    if accuracy:
        plots.accuracy(pos_pred, neg_pred)
    neg_preds = neg_pred.view(-1, 20)

    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_preds,
    })['mrr_list'].mean().item(), pos_pred, neg_pred


def runNN(epochs, load, save, batchsize=None, runs=1, plot=True, explain=False):
    dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = dataset.load(transform=True)
    rep = torch.from_numpy(np.asarray(pd.read_csv("data/baseline_NN")))

    # initilize models
    nn = Baseline()
    if batchsize is None:
        batchsize = dataset.num_edges
    if explain:
        # to plot proper plot the the LRP-method we need all walks:
        explain_data = dataset.load(transform=True, explain=True)
        # exp_adj = utils_func.adjMatrix(explain_data.edge_index,explain_data.num_nodes)  # Transpose of adj Matrix
        # for find walks walks uses rows as citing instance
    print(dataset.num_edges)
    if load:
        nn.load_state_dict(torch.load("model/nn_baseline_None_50_001_Nadam"))

    nn.to(device), data.to(device)
    optimizer = torch.optim.Adam(list(nn.parameters()), lr=0.0005)

    evaluator = Evaluator(name='ogbl-citation2')

    # ----------------------- training & testing
    average = np.zeros((runs, 2))
    for run in range(runs):
        valid_mrr, test_mrr, loss = torch.zeros((epochs, 1)), torch.zeros((epochs, 1)), torch.zeros((epochs, 1))
        old = 0
        pred_pos = np.zeros((epochs, test_set["target_node"].shape[0]))
        pred_neg = np.zeros((epochs, 9840))
        for i in range(0, epochs):
            print(i)
            if save:
                loss[i] = train(optimizer, train_set, rep, data.x, nn, batchsize).detach()
            valid_mrr[i], pred_pos[i, :], pred_neg[i, :] = test(evaluator, valid_set, nn, rep, data.x, batchsize)
            test_mrr[i], pred_pos[i, :], pred_neg[i, :] = test(evaluator, test_set, nn, rep, data.x, batchsize)

            if valid_mrr[i] > old and save:
                old = valid_mrr[i]
                tmp_nn = copy.deepcopy(nn.state_dict())

            if i == epochs - 1:
                if save:
                    torch.save(tmp_nn, "model/nn_baseline_None_50_001_80_stuff")
                if plot:
                    plots.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                                      ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error',
                                      file_name="NN_" + "performance_10")
                if explain:
                    explains(test_set, nn, explain_data.x, explain_data.adj_t, rep)
        # utils_func.accuracy_overtrain(pred_pos, pred_neg, epochs)

        average[run, 0] = valid_mrr[-1]
        average[run, 1] = test_mrr[-1]
    print("Testset avarage Performance:", average[:, 1].mean(), "Testset variance:",
          ((average[:, 1] - average[:, 1].mean()) ** 2 / runs).sum())
    print("Validation avarage Performance:", average[:, 0].mean(), "Validation variance:",
          ((average[:, 0] - average[:, 0].mean()) ** 2 / runs).sum())


def main():
    # run all baseline and return results
    runNN(epochs=1, load=True, save=False, batchsize=None, runs=1, plot=True)
    dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)
    data = dataset.load()
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    evaluator = Evaluator(name='ogbl-citation2')

    print(run_cn(evaluator, test_set, data.adj_t.to_dense()))
    print(run_cn(evaluator, valid_set, data.adj_t.to_dense()))


#    create_Dataset(data.adj.to_symmetric(),data.x)


if __name__ == "__main__":
    main()

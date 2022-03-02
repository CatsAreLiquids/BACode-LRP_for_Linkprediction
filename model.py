import copy

import numpy as np
import torch.nn
import torch_sparse
from ogb.linkproppred import Evaluator
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv

import dataLoader
import utils
import utils_func


class GNN(torch.nn.Module):
    """
    3-layer GNN with 128 input- and 256 output- and hidden neurons
    """

    def __init__(self):
        # build GNN here
        super(GNN, self).__init__()
        self.input = GCNConv(128, 256, bias=True)
        self.hidden = GCNConv(256, 256, bias=True)
        self.output = GCNConv(256, 256, bias=True)

    def forward(self, x, edge_index):
        h = self.input(x, edge_index)
        X = relu(h)

        h = self.hidden(X, edge_index)
        X = relu(h)

        h = self.output(X, edge_index)
        return h


    def lrp(self, x, edge_index, walk, r_src, r_tar, tar, epsilon=0.1, gamma=0.1):

        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.lin.weight[:, :] = cp.lin.weight + gamma * torch.clamp(cp.lin.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        A[0] = x
        A[0] = relu(A[0]).data.clone().requires_grad_(True)
        A[1] = relu(self.input(A[0], edge_index)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1], edge_index)).data.clone().requires_grad_(True)

        if walk[-1] == tar:
            R[-1] = r_tar
        else:
            R[-1] = r_src

        z = epsilon + roh(self.output).forward(A[2], edge_index)
        z = z[walk[2]]
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[2].grad
        R[1] = A[2] * c
        R[1] = R[1][walk[1]]

        z = epsilon + roh(self.hidden).forward(A[1], edge_index)
        z = z[walk[1]]
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c
        R[0] = R[0][walk[0]]

        """
        z = epsilon + roh(self.input).forward(A[0], edge_index)
        z = z[walk[1]]
        s = R[0] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[0].grad
        test = A[0] * c
        """
        return R[0]


class NN(torch.nn.Module):
    """
    3-Layer MLP with 256 input and hidden neurons and 1 output neuron
    """

    def __init__(self):
        # build MLP here
        super(NN, self).__init__()
        self.input = torch.nn.Linear(256, 256, bias=True)
        self.hidden = torch.nn.Linear(256, 256, bias=True)
        self.output = torch.nn.Linear(256, 1, bias=True)

    def forward(self, src, tar):
        x = src + tar
        h = self.input(x)
        X = relu(h)
        h = self.hidden(X)
        X = relu(h)
        h = self.output(X)
        return h

    # noinspection PyTypeChecker
    def lrp(self, src, tar, r, epsilon=0.1, gamma=0.1):
        def roh(layer):
            with torch.no_grad():
                cp = copy.deepcopy(layer)
                cp.weight[:, :] = cp.weight + gamma * torch.clamp(cp.weight, min=0)
                return cp

        A = [None] * 3
        R = [None] * 3

        R[-1] = r

        src = src.data.clone().requires_grad_(True)
        tar = tar.data.clone().requires_grad_(True)

        A[0] = relu(src + tar)
        A[0] = A[0].data.clone().requires_grad_(True)
        A[1] = relu(self.input(src + tar)).data.clone().requires_grad_(True)
        A[2] = relu(self.hidden(A[1])).data.clone().requires_grad_(True)

        z = epsilon + roh(self.output).forward(A[2])
        s = R[2] / (z + 1e-15)
        (z * s.data).sum().backward()

        c = A[2].grad
        R[1] = A[2] * c
        print("R",R[1].sum())

        z = epsilon + roh(self.hidden).forward(A[1])
        print(z.shape)
        s = R[1] / (z + 1e-15)
        (z * s.data).sum().backward()
        c = A[1].grad
        R[0] = A[1] * c

        """
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
        print(R[0].sum(),R[1].sum(),R[2].sum())
        return src * src_grad, tar * tar_grad


def train(batchsize, train_set, x, adj, optimizer, gnn, nn):
    # generating random permutation
    permutation = torch.randperm(train_set["source_node"].shape[0])
    total_loss = []
    num_sample = 0

    for i in range(0, train_set["source_node"].shape[0], batchsize):
        # TODO which optimizer
        optimizer.zero_grad()
        # Set up the batch
        idx = permutation[i:i + batchsize]
        train_src, train_tar = train_set["source_node"][idx], train_set["target_node"][idx]

        # removing positive link for training
        tmp = adj.to_dense()
        tmp[train_src, train_tar] = 0
        tmp[train_tar,train_src] = 0
        tmp = torch_sparse.SparseTensor.from_dense(tmp)
        graph_rep = gnn(x, tmp)

        # positive sampling
        out = torch.sigmoid(nn(graph_rep[train_src], graph_rep[train_tar]))
        pos_loss = - torch.mean(torch.log(out + 1e-15))

        neg_tar = torch.randint(low=0, high=22064, size=train_src.size(), dtype=torch.long)  # 30657
        out = torch.sigmoid(nn(graph_rep[train_src], graph_rep[neg_tar]))
        neg_loss = - torch.mean(torch.log(1 - out + 1e-15))

        # compute error
        loss = pos_loss + neg_loss

        # backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
        num_sample += batchsize
        # print("Num of samples done: ", num_sample, "/", train_set["source_node"].shape[0], " batch loss :", loss)

    return sum(total_loss) / num_sample


def explains(test_set, gnn, mlp, adj, x,edge_index):
    src, tar = test_set["source_node"], test_set["target_node"]
    walks_all = utils.walks(adj)
    print(adj)
    # forward passes
    mid = gnn(x, edge_index)  # features, edgeindex
    pos_pred = mlp(mid[src], mid[tar])
    samples = [8, 107, 14, 453]

    for i in samples:
        walks = utils_func.find_walks(src[i], tar[i], walks_all)
        r_src, r_tar = mlp.lrp(mid[src[i]], mid[tar[i]], pos_pred[i])
        #print(r_tar,r_src)
        p = []
        for walk in walks:
            p.append(gnn.lrp(x, edge_index, walk, r_src, r_tar, tar[i]))
        utils_func.plot_explain(p, src[i], tar[i], walks, "pos", i)


@torch.no_grad()
def test(batchsize, data_set, x, adj, evaluator, gnn, nn,accuracy=False):
    tmp = data_set["source_node"].shape[0]
    permutation = torch.randperm(tmp)
    src, tar, tar_neg = data_set["source_node"], data_set["target_node"], data_set["target_node_neg"]

    pos_preds, neg_preds = [], []
    graph_rep = gnn(x, adj)
    for i in range(0, src.shape[0], batchsize):
        idx = permutation[i:i + batchsize]
        src_tmp = src[idx]
        tar_tmp = tar[idx]
        tar_neg_tmp = tar_neg[idx]

        # positive sampling
        pos_preds += [nn(graph_rep[src_tmp], graph_rep[tar_tmp]).squeeze().cpu()]

        # negative sampling
        src_tmp = src_tmp.view(-1, 1).repeat(1, 20).view(-1)
        tar_neg_tmp = tar_neg_tmp.view(-1)
        neg_preds += [nn(graph_rep[src_tmp], graph_rep[tar_neg_tmp]).squeeze().cpu()]

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0)
    if accuracy : utils_func.accuracy(pos_pred,neg_pred)
    neg_pred = neg_pred.view(-1, 20)


    return evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()


def main(batchsize=None, epochs=1, explain=False, save=False, load=True, runs=100, plot=True):
    # ----------------------- Set up
    # globals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.default_rng()

    # loading the data
    dataset = dataLoader.LinkPredData("data/", "mini_graph", use_small=True)

    data = dataset.load()
    split = dataset.get_edge_split()
    train_set, valid_set, test_set = split["train"], split["valid"], split["test"]

    data.adj_t.to_symmetric()
    # Pre-compute GCN normalization.
    adj_t = data.adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    data.adj_t = adj_t

    # TODO how to degree matrix lets do out for now
    """
    tmp = data.adj_t.set_diag()
    deg = tmp.sum(dim=0).pow(-0.5)
    deg[deg == float('inf')] = 0
    tmp = deg.view(-1, 1) * tmp * deg.view(1, -1)
    data.adj_t = tmp
    """
    """
    # manipulating train for mrr computation
    permutation = torch.randperm(int(np.array(train_set["source_node"].shape[0])))[0:126]
    mrr_train = {"source_node": train_set["source_node"][permutation],
                 "target_node": train_set["target_node"][permutation],
                 "target_node_neg": valid_set["target_node_neg"]
                 }
    """

    # initilaization models
    gnn, nn = GNN(), NN()
    if load:
        gnn.load_state_dict(torch.load("model/gnn_None_50_01"))
        nn.load_state_dict(torch.load("model/nn_None_50_01"))
    gnn.to(device), nn.to(device), data.to(device)
    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(nn.parameters()),lr=0.01) #redo 2100 0.005
    evaluator = Evaluator(name='ogbl-citation2')

    # adjusting batchsize for full Dataset
    if batchsize is None:
        batchsize = dataset.num_edges
    if explain:
        # to plot proper plot the the LRP-method we need all walks:
        explain_data = dataset.load(transform=False, explain=True)
        exp_adj = utils_func.adjMatrix(explain_data.edge_index,explain_data.num_nodes)  # Transpose of adj Matrix for find walks
        # walks uses rows as citing instance

    # ----------------------- training & testing
    average = np.zeros((runs, 2))
    for run in range(runs):
        valid_mrr, test_mrr, loss = torch.zeros((epochs, 1)), torch.zeros((epochs, 1)), torch.zeros((epochs, 1))
        old = 0
        for i in range(0, epochs):
            print(i)
            if save:
                loss[i] = train(batchsize, train_set, data.x, data.adj_t, optimizer, gnn, nn).detach()

            test_mrr[i] = test(batchsize, test_set, data.x, data.adj_t, evaluator, gnn, nn)
            valid_mrr[i] = test(batchsize, valid_set, data.x, data.adj_t, evaluator, gnn, nn)

            if valid_mrr[i] > old and save:
                old = valid_mrr[i]
                tmp_gnn = copy.deepcopy(gnn.state_dict())
                tmp_nn = copy.deepcopy(nn.state_dict())

            if i == epochs - 1:
                if save:
                    torch.save(tmp_gnn, "model/gnn_None_50_01")
                    torch.save(tmp_nn, "model/nn_None_50_01")
                if plot:
                    utils_func.plot_curves(epochs, [valid_mrr, test_mrr, loss],
                                           ["Valid MRR", "Test MRR", "Trainings Error"], 'Model Error',
                                           file_name="GNN" + "performance.pdf")
                if explain:
                    explains(valid_set, gnn, nn, exp_adj, explain_data.x,explain_data.edge_index)

        average[run, 0] = valid_mrr[-1]
        average[run, 1] = test_mrr[-1]
    print("Testset avarage Performance:", average[:, 1].mean(), "Testset variance:",
          ((average[:, 1] - average[:, 1].mean()) ** 2 / runs).sum())
    print("Validation avarage Performance:", average[:, 0].mean(), "Validation variance:",
          ((average[:, 0] - average[:, 0].mean()) ** 2 / runs).sum())


if __name__ == "__main__":
    main()

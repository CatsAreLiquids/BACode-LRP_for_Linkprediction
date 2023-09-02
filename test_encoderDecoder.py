#%%
import torch_sparse
import unittest
from encoderDecoder import NN, GNN, testGCN
import torch
import dataLoader
from utils import utils_func
from encoderDecoder import refactored_explains, get_single_node_adjacency

class test_lrp(unittest.TestCase):

    def setUp(self) -> None:
        # loading the data
        self.dataset = dataLoader.LinkPredData("data/", "mini_graph", use_subset=True)

        self.data = self.dataset.load()
        split = self.dataset.get_edge_split()
        self.train_set, self.valid_set, self.test_set = split["train"], split["valid"], split["test"]

        self.nn = NN()
        self.gnn = GNN()
        self.gnn.load_state_dict(torch.load("models/gnn_2100_50_0015"))
        self.nn.load_state_dict(torch.load("models/nn_2100_50_0015"))
        self.t_GCN = testGCN(self.gnn)
        print('setup done')

    # def test_explains(self):
    #     explain_data = self.dataset.load(transform=False, explain=False)
    #     exp_adj = utils_func.adjMatrix(explain_data.edge_index,
    #                                    explain_data.num_nodes)  # Transpose of adj Matrix for find walks
    #     test_set = self.valid_set; gnn = self.gnn; mlp = self.nn; adj = exp_adj
    #     x = explain_data.x; edge_index = self.data.adj_t; validation_plot = False

    #     result = refactored_explains(test_set, gnn, mlp, adj, x, edge_index, validation_plot)
    
    def test_get_single_node_adjacency(self):
        onehot = lambda x,y : torch.tensor([0 if i != x else 1 for i in range(y)])
        top = 5
        features = torch.stack([onehot(x, top) for x in range(top)])
        adj = torch.tensor([
            [1,1,0,0,0],
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,1],
            [1,0,0,0,1],
        ])
        adj_sparse = torch_sparse.SparseTensor.from_dense(adj)
        for i in range(top):
            assert ((adj@features)[i] == (get_single_node_adjacency(adj_sparse, i)@features)[i]).all()

    def tearDown(self) -> None:
        print('teardown')
#%%
if __name__ == '__main__':
    unittest.main()
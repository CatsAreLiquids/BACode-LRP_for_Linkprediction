#%%
import numpy as np
import pandas as pd
import torch
from torch_geometric import data, transforms
from typing import Dict

class ToyData:
    def __init__(self, *args, **kwargs) -> None:
        self.transform = transforms.ToSparseTensor()

    def load(self, transform= True, explain = False) -> data.Data:
        # self.edge_index = torch.tensor([[0, 1, 2, 3, 5, 1, 4, 1, 4, 4],
        #                            [1, 4, 1, 4, 4, 0, 1, 2, 3, 5]])
        self.edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 5],
                                        [1, 4, 1, 4, 1, 4, 3, 2]])
        x = torch.tensor([[1,0]*64,
                          [1,0]*64,
                          [1,0]*64,
                          [0,1]*64,
                          [0,1]*64,
                          [0,1]*64]).to(torch.float32)
        dataset = data.Data(x, self.edge_index)
        self.num_edges = self.edge_index.shape[1]
        self.num_nodes = x.shape[0]

        if transform: dataset = self.transform(dataset)
        return dataset
    
    def get_edge_split(self) -> Dict[str,Dict[str, torch.Tensor]]:
        source_node = self.edge_index[0]
        target_node = self.edge_index[1]
        target_node_neg = (self.edge_index[1] + 3)%6
        target_node_neg[(target_node_neg == 1) | (target_node_neg == 4)] -= 1

        

        valid = test = train = {'source_node' : source_node, 'target_node' : target_node, 'target_node_neg': target_node_neg}

        split = {"train": train, "valid": valid, "test": test}
        return split
        

class LinkPredData:
    def __init__(self, root_dir, name, transform=transforms.ToSparseTensor(), use_year=False, use_subset=True,):

        self.root_dir = root_dir
        self.name = name
        self.transform = transform
        self.subset = use_subset
        self.use_year = use_year

    def load(self, transform=True, explain=False):
        if self.subset:
            if explain:
                edges = pd.read_csv(self.root_dir + self.name + "_edges_indexed").to_numpy()
            else:
                edges = pd.read_csv(self.root_dir + self.name + "_edges_train").to_numpy()
            features = pd.read_csv(self.root_dir + self.name + "_features").to_numpy()
        else:
            edges = pd.read_csv(self.root_dir + "edge.csv", header=None).to_numpy()
            features = pd.read_csv(self.root_dir + "node-feat.csv", header=None).to_numpy()

        edge_index = torch.from_numpy(edges.T)
        x = torch.from_numpy(features)
        x = x.to(torch.float32)  # more for consistency than necessity
        dataset = data.Data(x, edge_index)
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]

        if transform:
            return self.transform(dataset)
        else:
            return dataset

    def get_edge_split(self):

        if self.subset:
            valid = torch.load(self.root_dir + "mini_graph_valid.pt")
            test = torch.load(self.root_dir + "mini_graph_test.pt")
            train = torch.load(self.root_dir + "mini_graph_train.pt")
        else:
            test = torch.load(self.root_dir + "test.pt")
            train = torch.load(self.root_dir + "train.pt")
            valid = torch.load(self.root_dir + "valid.pt")

        for key in valid.keys():
            valid[key] = torch.from_numpy(np.array(valid[key]))
        for key in test.keys():
            test[key] = torch.from_numpy(np.array(test[key]))
        for key in train.keys():
            train[key] = torch.from_numpy(np.array(train[key]))
        split = {"train": train, "valid": valid, "test": test}

        return split

    def get_year(self):
        if self.subset:
            year_path = self.root_dir + self.name + "_year"
        else:
            year_path = self.root_dir + "node_year.csv"
        year = pd.read_csv(year_path).to_numpy()
        return year

    def get_representation(self, baseline):
        rep = torch.from_numpy(np.asarray(pd.read_csv(self.root_dir + baseline)))
        return rep

if __name__ == '__main__':
    dataset = LinkPredData("data/", "mini_graph", use_subset=True)
    d = dataset.load()
    split = dataset.get_edge_split()
    print(d)
    print(split)
    ei = torch.stack([split["test"]["source_node"], split["test"]["target_node"]])
# %%

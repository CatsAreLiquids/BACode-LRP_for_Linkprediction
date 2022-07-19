from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import DataLoader

# Download and process data at './dataset/ogbl-citation2/'
dataset = PygLinkPropPredDataset(name="ogbl-citation2", root='dataset/')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
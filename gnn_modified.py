import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from ogb.linkproppred import  Evaluator

import logging

import dataLoader
import utils_func
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i + x_j               # Modified from x_i * x_j to x_i + x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.relu(x)     # Modified from torch.sigmoid to torch.nn.ReLU(x)


def train(model, predictor, data, split_edge, optimizer, batch_size,adj_t):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data, adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        print(neg_out.size(),pos_out.size())
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size,adj_t):
    predictor.eval()

    h = model(data, adj_t)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    dataset,split_edge = dataLoader.main(full_dataset=True,use_year=False)
    data = dataset["x"]

    print("edges over all",dataset["edge_index"].size())

    print("training edges",split_edge["train"]["source_node"].size())
    print()
    adj_t = utils_func.adjMatrix(dataset["edge_index"],data.shape[0])



    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }
    model = GCN(data.shape[1], args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
    data = data.to(device),

    # Pre-compute GCN normalization.
    #TODO not the most efficient way

    deg = adj_t.sum(axis=1).to(torch.float)

    deg_inv_sqrt = deg ** (-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_t = np.matmul(np.matmul(deg_inv_sqrt,adj_t),deg_inv_sqrt) #originaly deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    adj_t = torch.from_numpy(adj_t).to(torch.long).to_sparse()
    adj_t = adj_t.to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation2')

    print("test")
    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)
        logger = logging.getLogger("GCN")
        logger.basicConfig(level=logging.INFO)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size,adj_t)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, data, split_edge, evaluator,
                              args.batch_sizeq,adj_t)
                logger.info(run)
                logger.info(result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')
"""
        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()
"""

if __name__ == "__main__":
    main()

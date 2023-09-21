#%%
import torch
import tqdm
import dataLoader
from utils import utils_func, ainb
import glob
import os
import pandas as pd


def compare_relevances(a : torch.Tensor, b : torch.Tensor, comparisson : str):
    if comparisson == "sum":
        a,b = a.sum(), b.sum()
    elif comparisson == "mean":
        a,b = a.mean(), b.mean()
    elif comparisson == "median":
        a,b = a.median(), b.median()

    return a,b 
        


def filter_relevances_and_pool(rel_matrix : torch.Tensor, src : torch.Tensor, tar : torch.Tensor):
    all_walks = rel_matrix._indices()
    relevances = rel_matrix._values()

    contains_src = (all_walks == src).any(dim=0)
    contains_tar = (all_walks == tar).any(dim=0)

    is_indirect_path = contains_src & contains_tar

    indirect_path_relevance = relevances[is_indirect_path]
    no_path_relevance = relevances[~is_indirect_path]

    connected_dict = {}
    not_connected_dict = {}

    for comparisson in ["sum", "mean", "median"]:
        connected, no_path = compare_relevances(indirect_path_relevance, no_path_relevance, comparisson)
        connected_dict["connected_" + comparisson] = connected.item()
        not_connected_dict["not_connected_" + comparisson] = no_path.item()
    
    return connected_dict, not_connected_dict 

def get_pooled_relevances(path_to_folder : str, all_src : torch.Tensor, all_tar : torch.Tensor):
    out = []
    for src, tar in tqdm.tqdm(zip(all_src, all_tar), desc= "Source-Target pairs:", total= len(all_src)):
        files = glob.glob(os.path.join(path_to_folder,f"{src.item()}_{tar.item()}_*.th"))
        d = {}
        for file in files:
            rel_matrix : torch.Tensor = torch.load(file)
            filename = os.path.splitext(file)[0].split('/')[-1]
            _src, _tar, gamma, epsilon = filename.split('_')
            gamma, epsilon = [float(num.replace(',', '.')) for num in [gamma, epsilon]]
            connected_dict, not_connected_dict = filter_relevances_and_pool(rel_matrix, src, tar)
            d = {
                'src' : src.item(),
                'tar' : tar.item(),
                'gamma' : gamma,
                'epsilon' : epsilon,
                'total_rel' : rel_matrix.sum().item(),
            }
            d.update(connected_dict); d.update(not_connected_dict)
        if len(d) > 0:
            d = pd.Series(d)
            out.append(d)

    return pd.DataFrame(out)


def sp(tensor, indices, iteration= 0):
    if len(indices) == 0:
        return tensor
    if len(indices) == 1:
        return tensor.index_select(iteration, torch.tensor([indices[0]]))._values()
    else:
        return sp(tensor.index_select(iteration, torch.tensor([indices[0]])), indices[1:], iteration + 1)
# %%

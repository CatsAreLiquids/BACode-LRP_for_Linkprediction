import torch
import torch_sparse
import pickle
from typing import Dict, List, Union, Literal
from tqdm import tqdm
from . import utils_func, ainb

def find_good_samples(adj : torch_sparse.SparseTensor, subset : Dict[str, Union[torch.Tensor, List[torch.Tensor]]], criterion : Literal['indirect connections'] = None, remove_connection : bool = True, load : str = "", save : bool = False, **kwargs) -> Dict[int, List[List[int]]]:
    """
    Finds good samples to analyze based on a given criterion

    @param: load: If load is not an empty string, try "torch.load(load)". If it fails continue with criterion
    @param: save: If True, then store a pickle file with the computed walks
    """
    if load != "":
        try:
            with open(load, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError as err:
            print(f"Loadig from {load} failed")
            print(err)
            print(f"Continuing with criterion: {criterion}")
    if criterion is None: return [5, 47 , 53, 5, 188, 105]
    if criterion == "indirect connections":
        out = _find_samples_with_indirect_connections(adj, subset, remove_connection)

    if save:
        if type(save) is str:
            save_path = save
        else:
            save_path = f"walks with {criterion}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(out, f)
    return out


def remove_connections_between(_adj: torch_sparse.SparseTensor, node_a: Union[torch.Tensor, int], node_b: Union[torch.Tensor, int]) -> torch_sparse.SparseTensor:
    adj = _adj.clone()
    if isinstance(node_a, torch.Tensor) and len(node_a) > 1:
        for _a, _b in zip(node_a, node_b):
            _a = _a.item() if isinstance(_a, torch.Tensor) else _a
            _b = _b.item() if isinstance(_b, torch.Tensor) else _b
            adj = remove_connection_at_index(adj, find_index_of_connection(adj, _a, _b))

    else:
        node_a = node_a.item() if isinstance(node_a, torch.Tensor) else node_a
        node_b = node_b.item() if isinstance(node_b, torch.Tensor) else node_b
        adj = remove_connection_at_index(adj, find_index_of_connection(adj, _a, _b))
    return adj

def _find_samples_with_indirect_connections(adj : torch_sparse.SparseTensor, subset : Dict[str, torch.Tensor], remove_connection : bool = True) -> Dict[int, List[List[int]]]:
    out = {}
    for i, (src, tar, neg_tar) in tqdm(enumerate(zip(*subset.values())), desc= "Alternative connections between src and tar", total= len(subset["source_node"])):
        if remove_connection:
            _adj = remove_connection_at_index(adj, find_index_of_connection(adj, src, tar))
        else: _adj = adj
        walks = utils_func.walks(_adj, src, tar)
        indirect_connection_walks = [int(i) for i,w in enumerate(walks) if src.item() in w and tar.item() in w]
        if len(indirect_connection_walks) > 0:
            out[i] = [walks, indirect_connection_walks]
    return out

def find_index_of_connection(sparse_tensor : torch_sparse.SparseTensor, src : int, tar : int) -> torch.Tensor:
    """
    Finds the index in the sparse storage for a given connection, returns the indexes in a Tensor
    """
    rows = sparse_tensor.storage.row()
    cols = sparse_tensor.storage.col()
    _rows = torch.where(rows == tar)[0]
    _cols = torch.where(cols == src)[0]
    _rows_t = torch.where(rows == src)[0]
    _cols_t = torch.where(cols == tar)[0]
    idx = ainb.ainb(_rows, _cols)
    idx_t = ainb.ainb(_rows_t, _cols_t)
    assert idx.sum() <= 1 and idx_t.sum() <= 1, "Should find one match at most" # In the case of repeated indexes we are most likely seeing a runtime error
    remove_indexes = []
    if idx.sum(): remove_indexes.append(_rows[idx])
    if idx_t.sum(): remove_indexes.append(_rows_t[idx_t])
    remove_indexes.sort()
    assert torch.tensor(
        [[rang[r_idx] in [src, tar] for r_idx in remove_indexes] for rang in [rows, cols]]
        ).all() #  Checks that the indexes to be removed actually correspond to src and target
    
    return remove_indexes

def _find_index_of_neighbours(sparse_tensor : torch_sparse.SparseTensor, node : Union[int, torch.Tensor], direction : Literal['forward', 'backward'] = 'backward') -> torch.Tensor:
    prev_node, next_node, val = sparse_tensor.coo()

    if direction == "forward":    
        inds = torch.where(prev_node == node)[0]
    elif direction == "backward":
        inds = torch.where(next_node == node)[0]
    else:
        raise ValueError("Direction not recognized")
    
    inds = inds[val[inds] != 0] 

    return inds

def get_single_node_adjacency(sparse_tensor : torch_sparse.SparseTensor, node : Union[int, torch.Tensor], direction : Literal['forward', 'backward'] = 'backward') -> torch_sparse.SparseTensor:
    rows, cols, val = sparse_tensor.coo()

    if direction == "forward":    
         idx = _find_index_of_neighbours(sparse_tensor, node, 'forward')
    elif direction == "backward":
        idx = _find_index_of_neighbours(sparse_tensor, node, 'backward')
    else:
        raise ValueError("Direction not recognized")
    
    return torch_sparse.SparseTensor(row= rows[idx], col= cols[idx], value= val[idx], sparse_sizes=sparse_tensor.storage.sparse_sizes())

def remove_connection_at_index(sparse_tensor : torch_sparse.SparseTensor, indexes : list[int]) -> torch_sparse.SparseTensor:
    """
    Removes the connection at the indexes in the sparse storage, returns a new SparseTensor without the connections
    """
    if not len(indexes) > 0: return sparse_tensor # If connection is already missing just return same tensor
    new_row, new_col, new_value = [
        torch.cat([inner_tensor[i+1: j] for i,j in zip([-1] + indexes, indexes + [None])]) # Adding -1 at the beginning of the indexes means start from index 0
        for inner_tensor in [
                            sparse_tensor.storage.row(),
                            sparse_tensor.storage.col(),
                            sparse_tensor.storage.value()
                            ]
    ]
    return torch_sparse.SparseTensor(new_row, None, new_col, new_value, sparse_tensor.sizes())
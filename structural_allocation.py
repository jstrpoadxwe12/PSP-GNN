import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_self_loops, to_dense_adj


def knn_graph(K, matrix):
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)

    matrix = matrix - torch.diag_embed(torch.diag(matrix))
    _, indices = matrix.topk(k=int(K), dim=-1)

    knn_edge_index = torch.stack((torch.flatten(indices),
                                  torch.repeat_interleave(torch.arange(matrix.shape[0]), K)))

    knn_edge_index = to_undirected(knn_edge_index) 
    knn_edge_index, _ = remove_self_loops(knn_edge_index)
    return knn_edge_index


def calc_entropy(adj: torch.Tensor):
    adj = adj - torch.diag_embed(torch.diag(adj))
    degree = adj.sum(dim=1)
    vol = adj.sum()
    idx = degree.nonzero().reshape(-1)
    g = degree[idx]
    return -((g / vol) * torch.log2(g / vol)).sum()


def edge_index_to_adj(node_num, matrix, edge_index):
    weight = torch.tensor(matrix[edge_index[0], edge_index[1]])
    return to_dense_adj(edge_index, None, weight, node_num)[0]


def structural_allocation_strategy(matrix, threshold=0.001):
    old_entropy = 0
    node_num = matrix.shape[0]
    k = 1
    while k < node_num:
        edge_index_k = knn_graph(k, matrix)
        adj = edge_index_to_adj(node_num, matrix, edge_index_k)
        new_entropy = calc_entropy(adj)

        if new_entropy - old_entropy > threshold:
            k += 1
        else:
            break
        old_entropy = new_entropy
    return k

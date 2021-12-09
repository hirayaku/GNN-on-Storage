import torch

from dgl.transform import metis_partition
from dgl import backend as F
import dgl

def get_partition_list(g, psize, mask=None):
    '''
    return a list of clusters; each item in the list is a graph nid array belonging to cluster i
    '''
    p_gs = metis_partition(g, psize, balance_ntypes=mask)
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        # NOTE: keep nids in ascending order
        nids, _ = torch.sort(nids)
        graphs.append(nids)
    return graphs

def get_partition_nodes(par_arr, i, psize, batch_size):
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    return torch.cat(par_batch_ind_arr)

def get_subgraph(g, par_arr, i, psize, batch_size):
    return g.subgraph(get_partition_nodes(par_arr, i, psize, batch_size))

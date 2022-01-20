import torch

from dgl.transform import metis_partition
from dgl.transform import edge_partition
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

def get_edge_partition_list(g, psize):
    '''
    return a list of vertex sets, each of which belongs to an edge partition and could overlap
    '''
    par_list = edge_partition(g, psize, "bidirected")
    return par_list
    # replicates = torch.zeros(g.num_nodes())
    # for par in par_list:
    #     replicates[par] += 1
    # return par_list, replicates

def get_rand_partition_list(g, psize):
    '''
    return a random partitioning of nodes
    '''
    nodes = g.nodes()[torch.randperm(g.num_nodes())]
    pcount = g.num_nodes() // psize
    par_list = [nodes[i * pcount : (i+1) * pcount] for i in range(psize-1)]
    par_list.append(nodes[(psize-1) * pcount :])
    return par_list

def get_partition_nodes(par_arr, i, psize, batch_size):
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    return torch.cat(par_batch_ind_arr)

def get_subgraph(g, par_arr, i, psize, batch_size):
    return g.subgraph(get_partition_nodes(par_arr, i, psize, batch_size))

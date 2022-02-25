import torch

from dgl.transform import metis_partition, metis_partition_assignment
from dgl.partition import ne_vertex_partition, NePolicy
import dgl
import dgl.function as fn

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

def get_nev_partition_list(g, psize, mode=NePolicy.NE_BALANCED):
    '''
    return a list of vertex sets, each of which belongs to an edge partition and could overlap
    '''
    train_g = g.subgraph(g.nodes()[g.ndata['train_mask'].bool()])
    seed_nids = train_g.ndata[dgl.NID]
    assignments = metis_partition_assignment(train_g, psize)
    # seed_nids = g.nodes()[g.ndata['train_mask'].bool()]
    # assignments = torch.randint(psize, seed_nids.size())
    par_list = ne_vertex_partition(g, psize, seed_nids, assignments, "bidirected", mode)
    return par_list

def get_nev_partition_list_randseed(g, psize, mode=NePolicy.NE_BALANCED):
    '''
    return a list of vertex sets, each of which belongs to an edge partition and could overlap
    '''
    seed_nids = g.nodes()[g.ndata['train_mask'].bool()]
    assignments = torch.randint(psize, seed_nids.size())
    par_list = ne_vertex_partition(g, psize, seed_nids, assignments, "bidirected", mode)
    return par_list

def get_rand_partition_list(g, psize):
    '''
    return a random partitioning of nodes
    '''
    nodes = g.nodes()[torch.randperm(g.num_nodes())]
    pcount = g.num_nodes() // psize
    par_list = [nodes[i * pcount : (i+1) * pcount] for i in range(psize-1)]
    par_list.append(nodes[(psize-1) * pcount :])
    return par_list

def get_rand_partition_list_clusterseed(g, psize):
    '''
    return a random partitioning of nodes
    '''
    par_list = get_partition_list(g.subgraph(g.nodes()[g.ndata['train_mask']]), psize)
    nontrain_nodes = g.nodes()[torch.logical_not(g.ndata['train_mask'])]
    nontrain_nodes = nontrain_nodes[torch.randperm(nontrain_nodes.size()[0])]
    pcount = nontrain_nodes.size()[0] // psize
    nontrain_par_list = [nontrain_nodes[i * pcount : (i+1) * pcount] for i in range(psize-1)]
    nontrain_par_list.append(nontrain_nodes[(psize-1) * pcount :])
    for i in range(psize):
        par_list[i] = torch.cat((par_list[i], nontrain_par_list[i]), dim=0)
    return par_list


def get_partition_nodes(par_arr, i, psize, batch_size):
    par_batch_ind_arr = [par_arr[s] for s in range(
        i * batch_size, (i + 1) * batch_size) if s < psize]
    return torch.cat(par_batch_ind_arr)

def get_subgraph(g, par_arr, i, psize, batch_size):
    return g.subgraph(get_partition_nodes(par_arr, i, psize, batch_size), )

def rescale(g, part_list, bsize):
    g.ndata['part'] = torch.zeros(g.num_nodes())
    for pn, nodes in enumerate(part_list):
        g.ndata['part'][nodes] = pn

    # assign rescale coefficients to each edge
    rescale_coeff = (len(part_list) - 1) / (bsize - 1)
    g.apply_edges(fn.u_sub_v('part', 'part', 'rs'))
    g.edata['rs'] = (g.edata['rs'] != 0).float() * rescale_coeff + \
        (g.edata['rs'] == 0).float()
    
    # assign a normalization factor to each node
    norm = 1. / g.in_degrees().float().unsqueeze(1)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm

def partition_cuts_1hop(g, par_list, training_mask):
    g.ndata['part'] = torch.zeros(g.num_nodes())
    for pn, nodes in enumerate(par_list):
        g.ndata['part'][nodes] = pn
    g.apply_edges(fn.u_sub_v('part', 'part', 'part_diff'))
    g.edata['part_diff'] = (g.edata['part_diff'] != 0).float()
    # for nodes u, v, if they are in different partitions, e(u, v) = 1
    g.update_all(fn.copy_e('part_diff', 'm'), fn.sum('m', 'cuts'))

    par_cuts = torch.zeros(len(par_list))
    par_sizes = torch.zeros(len(par_list))
    par_ts_sizes = torch.zeros(len(par_list))
    for pn, nodes in enumerate(par_list):
        par_sizes[pn] = nodes.size()[0]
        par_training_nodes = nodes[training_mask[nodes] != 0]
        par_ts_sizes[pn] = par_training_nodes.size()[0]
        par_cuts[pn] = torch.sum(g.ndata['cuts'][par_training_nodes])
    
    return par_sizes, par_ts_sizes, par_cuts

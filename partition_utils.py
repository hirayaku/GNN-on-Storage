import torch

from dgl.partition import metis_partition, metis_partition_assignment
import dgl
import dgl.function as fn

# should we make partition functions return assignments, not partition lists?

def get_partition_list(g, psize, mask=None, balance_edges=False):
    '''
    return a list of clusters; each item in the list is a graph nid array belonging to cluster i
    '''
    p_gs = metis_partition(g, psize, balance_ntypes=mask, balance_edges=balance_edges)
    parts = []
    # intra_edges = torch.zeros(psize).long()
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        # intra_edges[k] = val.num_edges()
        parts.append(nids)
    return parts

def get_rand_partition_list(g, psize, **kwargs):
    '''
    return a random partitioning of nodes
    '''
    nodes = g.nodes()[torch.randperm(g.num_nodes())]
    pcount = g.num_nodes() // psize
    par_list = [nodes[i * pcount : (i+1) * pcount] for i in range(psize-1)]
    par_list.append(nodes[(psize-1) * pcount :])
    return par_list

def get_rand_partition_list_clusterseed(g, psize, **kwargs):
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

'''
def get_nev_partition_list(g, psize, mode=NePolicy.NE_BALANCED):
    train_g = g.subgraph(g.nodes()[g.ndata['train_mask'].bool()])
    seed_nids = train_g.ndata[dgl.NID]
    assignments = metis_partition_assignment(train_g, psize)
    print("METIS partitioning of train_g completes")
    # seed_nids = g.nodes()[g.ndata['train_mask'].bool()]
    # assignments = torch.randint(psize, seed_nids.size())
    par_list = ne_vertex_partition(g, psize, seed_nids, assignments, "bidirected", mode)
    return par_list

def get_nev_partition_list_randseed(g, psize, mode=NePolicy.NE_BALANCED):
    seed_nids = g.nodes()[g.ndata['train_mask'].bool()]
    assignments = torch.randint(psize, seed_nids.size())
    par_list = ne_vertex_partition(g, psize, seed_nids, assignments, "bidirected", mode)
    return par_list
'''

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

import numpy as np
import scipy.sparse as sp
import time

def nev_norm_randomized(g, psize, iterations=5):
    train_g = g.subgraph(g.nodes()[g.ndata['train_mask'].bool()])
    seed_nids = train_g.ndata[dgl.NID]
    assigned = seed_nids.shape[0]
    assignments = metis_partition_assignment(train_g, psize)
    frontiers = seed_nids
    frontiers_list = []
    assignments_list = [assignments]

    with g.local_scope():
        g.ndata["part"] = -torch.ones(g.num_nodes()).int()
        g.ndata["part"][frontiers] = assignments.int()

        # BFS
        timer = time.time()
        # dgl keeps COO format to index edges
        frontier_edges = g.out_edges(seed_nids)
        next_frontiers = torch.sort(torch.unique(frontier_edges[1])).values
        num_frontier_edges = frontier_edges[0].shape[0]
        num_next_frontiers = next_frontiers.shape[0]
        print(f"Get next frontiers({num_next_frontiers}, {num_frontier_edges}):"
            f" {time.time()-timer:.2f}s")

        # build probablity matrix
        timer = time.time()
        frontier_parts = g.ndata["part"][frontier_edges[0]].numpy()
        assert not (frontier_parts == -1).any()
        next_frontiers_reindex = torch.bucketize(frontier_edges[1], next_frontiers).numpy()
        partition_adj = sp.csr_matrix((torch.ones(num_frontier_edges),
            (next_frontiers_reindex, frontier_parts)))
        assert partition_adj.shape[0] == num_next_frontiers
        print(f"Build probablity matrix {partition_adj.shape}: {time.time()-timer:.2f}s")
        print(f"Nonzeros: {partition_adj.count_nonzero()}")

        # check imbalance
        col_sum = partition_adj.sum(axis=0)
        col_sum[col_sum == 0] = 1
        mean = col_sum.mean()
        print(f"Imbalance factor: max/mean={col_sum.max()/mean:.2f}, "
            f"mean/min={mean/col_sum.min():.2f}")

        # start iteration
        desired_row_sum = 1
        desired_col_sum = desired_row_sum * num_next_frontiers / psize
        for k in range(iterations):
            timer = time.time()
            # row normalization then -> csc
            partition_adj = partition_adj.multiply(
                desired_row_sum / partition_adj.sum(axis=1))
            # col normalization then -> csr
            col_sum = partition_adj.sum(axis=0)
            col_sum[col_sum == 0] = 1
            partition_adj = partition_adj.multiply(
                desired_col_sum / partition_adj.sum(axis=0))
            print(f"Row&col norm, Iteration {k}: {time.time()-timer:.2f}s")

        # ends with a row norm
        partition_adj = partition_adj.multiply(
            desired_row_sum / partition_adj.sum(axis=1))
        # check imbalance
        col_sum = partition_adj.sum(axis=0)
        nzeros = (col_sum == 0).sum()
        col_sum[col_sum == 0] = 1
        mean = col_sum.mean()
        print(f"Number of non-increasing partitions: {nzeros}")
        print(f"Imbalance factor: max/mean={col_sum.max()/mean:.2f}, "
            f"mean/min={mean/col_sum.min():.2f}")

        frontiers = next_frontiers
        # choose assignments for each new frontier
        parts = [np.random.choice(psize, p=partition_adj.getrow(i).toarray().ravel())
            for i in range(num_next_frontiers)]
        assignments = torch.IntTensor(parts)
        assignments_list.append(torch.Tensor(parts))
        assigned += assignments.shape[0]
        g.ndata["part"][frontiers] = assignments.int()
        print(f"Assigned: {assigned/g.num_nodes()}")


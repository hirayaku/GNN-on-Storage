import os, random

import numpy as np
import torch, dgl
import dgl.function as fn

class NodePartitioner(object):
    def __init__(self, name):
        self.name = name

    def partition(self, g, psize):
        raise NotImplementedError

class RandomNodePartitioner(NodePartitioner):
    def __init__(self):
        super().__init__('rand')

    def partition(self, g, psize):
        return torch.randint(psize, (g.num_ndoes(),))

class MetisNodePartitioner(NodePartitioner):
    def __init__(self, name='metis'):
        super().__init__(name)

    def partition(self, g, psize, mask=None, objtype='cut'):
        return dgl.metis_partition_assignment(g, psize, mask, objtype=objtype)

class MetisMinCutBalanced(MetisNodePartitioner):
    def __init__(self):
        super().__init__(name='metis-cut')

    def partition(self, g, psize):
        return super().partition(g, psize, g.ndata['train_mask'].int(), objtype='cut')
    
class MetisMinVolBalanced(MetisNodePartitioner):
    def __init__(self):
        super().__init__(name='metis-vol')

    def partition(self, g, psize):
        return super().partition(g, psize, g.ndata['train_mask'].int(), objtype='vol')

def partition_from(ids, assigns, psize):
    '''
    return ids of each part in a list given the assignments
    '''
    assert ids.shape == assigns.shape
    _, idx = torch.sort(assigns)
    shuffled = ids[idx]

    # compute partition sizes
    group_sizes = torch.histc(assigns.float(), bins=psize, min=0, max=psize).long()
    group_offs = torch.cumsum(group_sizes, dim=0)
    groups = [shuffled[:group_offs[0]]]
    for i in range(1, len(group_offs)):
        groups.append(shuffled[group_offs[i-1]:group_offs[i]])

    return groups

def edge_cuts_from(g, assigns, psize):
    '''
    return #edge cuts between pairs of partitions
    u -[e]-> v: id(e) = id(u) * psize + id(v)
    '''
    assert g.num_nodes() == assigns.shape[0]
    with g.local_scope():
        g.ndata['vp'] = assigns.float()
        g.ndata['vp_p'] = g.ndata['vp'] * psize
        g.apply_edges(fn.u_add_v('vp', 'vp_p', 'ep'))
        edge_part_sizes = torch.histc(g.edata['ep'], bins=psize*psize, min=0, max=psize*psize)
        return edge_part_sizes.reshape((psize,psize))

def train_edge_cuts_from(g, assigns, psize):
    '''
    return #edge cuts between pairs of partitions
    u -[e]-> v: id(e) = id(u) * psize + id(v)
    '''
    assert g.num_nodes() == assigns.shape[0]
    train_mask = g.ndata['train_mask']
    with g.local_scope():
        g.ndata['t_p'] = -1 * torch.ones(g.num_nodes())
        g.ndata['t_p'][train_mask] = assigns[train_mask].float()
        g.ndata['t_p'] = g.ndata['t_p'] * psize
        g.ndata['nt_p'] = assigns.float()
        g.apply_edges(fn.u_add_v('t_p', 'nt_p', 'e_p'))
        edge_part_sizes = torch.histc(g.edata['e_p'], bins=psize*psize, min=0, max=psize*psize)
        return edge_part_sizes.reshape((psize,psize))

class ClusterIterV2(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis/other partitioners is used as the graph partition backend.
    The sampler returns a subgraph induced by a batch of clusters
    '''
    def __init__(self, dataset, g, psize, bsize, hsize, partitioner=RandomNodePartitioner(),
            sample_helpers=False, sample_topk=False, popular_ratio=0):
        self.g = g
        self.psize = psize
        self.bsize = bsize
        self.hsize = hsize
        self.sample_topk = sample_topk
        self.sample_helpers = sample_helpers

        cache_folder = os.path.join(os.environ['DATASETS'], "partition",
                dataset, partitioner.name)
        os.makedirs(cache_folder, exist_ok=True)
        cache_file = f'{cache_folder}/p{psize}.pt'

        nids = g.nodes()
        train_nids = nids[g.ndata['train_mask']]
        if os.path.exists(cache_file):
            self.assigns = torch.load(cache_file)
        else:
            self.assigns = partitioner(g, psize)
            torch.save(self.assigns, cache_file)
        nontrain_mask = ~g.ndata['train_mask']
        self.parts = partition_from(nids[nontrain_mask], self.assigns[nontrain_mask], psize)
        train_assigns = self.assigns[train_nids]
        self.train_parts = partition_from(train_nids, train_assigns, psize)
        self.train_nids = train_nids

        # print("Computing partition sampling weights...")
        # self.cuts = train_edge_cuts_from(g, self.assigns, psize)

        self.max = int((self.psize) // self.bsize)
        self.train_pids = torch.randperm(self.psize)

        self.node_cache_parts = None
        self.node_budget = int(popular_ratio * g.num_nodes())
        self.node_prob = self.g.in_degrees().float()
        self.node_prob = self.node_prob / self.node_prob.sum()
        self.node_cache_parts = self.__get_popular_nodes__()

    def __get_popular_nodes__(self):
        if self.node_cache_parts is None or self.sample_topk:
            if self.sample_topk:
                node_cache = torch.from_numpy(
                        np.random.choice(np.arange(self.g.num_nodes(), dtype=np.int64),
                            size=self.node_budget, p=self.node_prob.numpy(), replace=True)
                        )
            else:
                node_cache = torch.topk(self.node_prob, self.node_budget).indices
            return partition_from(node_cache, self.assigns[node_cache], self.psize)
        else:
            return self.node_cache_parts

    '''
    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['feat']
        with torch.no_grad():
            g.update_all(fn.copy_src(src='feat', out='m'),
                         fn.sum(msg='m', out='feat'),
                         None)
            pre_feats = g.ndata['feat'] * norm
            # use graphsage embedding aggregation style
            g.ndata['feat'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['feat'].device)
        return norm
    '''

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            sample_pids = self.train_pids[self.n*self.bsize : (self.n+1)*self.bsize]
            train_nids = torch.cat([self.train_parts[pid] for pid in sample_pids])
            # we use fixed helpers policy
            helper_nids = torch.cat([self.parts[i] for i in sample_pids])
            cache_parts = [self.node_cache_parts[i] for i in range(self.psize) if i not in sample_pids]
            cache_nids = torch.cat(cache_parts)
            nids = torch.cat([train_nids, helper_nids, cache_nids])
            # XXX: check cache_nids don't duplicate with other nodes
            assert len(torch.unique(nids)) == len(nids)

            subgraph = dgl.node_subgraph(self.g, nids)
            sg_num_nodes = subgraph.num_nodes()
            sg_train_nids = torch.arange(train_nids.shape[0])
            sg_cache_nids = torch.arange(sg_num_nodes-len(cache_nids), sg_num_nodes)
            # XXX: verify all sg_train_nids has train_mask
            assert (subgraph.ndata['train_mask'][sg_train_nids] == True).all().item()
            subgraph.ndata['train_mask'][:] = False
            subgraph.ndata['train_mask'][sg_train_nids] = True
            subgraph.ndata['cache_mask'] = torch.BoolTensor(sg_num_nodes)
            subgraph.ndata['cache_mask'][:] = False
            subgraph.ndata['cache_mask'][sg_cache_nids] = True
            self.n += 1
            return subgraph, sg_train_nids
        else:
            self.train_pids = torch.randperm(self.psize)
            self.node_cache_parts = self.__get_popular_nodes__()
            raise StopIteration

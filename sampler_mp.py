import os, time
import numpy as np
import torch, dgl
import dgl.function as fn

import getpy as gp
import utils, partition_utils
from partition_utils import partition_from

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
    def __init__(self, dataset, g, psize, bsize, hsize,
            partitioner=partition_utils.RandomNodePartitioner(),
            sample_helpers=False, sample_topk=False, popular_ratio=0):
        self.g = g
        self.psize = psize
        self.bsize = bsize
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
            self.assigns = partitioner.partition(g, psize)
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
            cache_nids = torch.cat(cache_parts) if len(cache_parts) > 0 else torch.LongTensor()
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

import gnnos
from gnnos_utils import store
from gnnos_graph import GnnosPartGraph
from graphloader import GnnosNodePropPredDataset
import torch.multiprocessing as mp

def lookup(gp_dict: gp.Dict, tensor: torch.Tensor):
    return torch.from_numpy(gp_dict[tensor.numpy()])

class GnnosIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis/other partitioners is used as the graph partition backend.
    The sampler returns a subgraph induced by a batch of clusters
    '''
    def __init__(self, gnnos_dataset: GnnosNodePropPredDataset, bsize):
        self.dataset = gnnos_dataset
        self.psize = self.dataset.psize
        self.bsize = bsize
        self.parts = self.dataset.parts
        self.pg = self.dataset.graph
        self.scache = self.dataset.scache
        self.scache_nids = self.dataset.scache_nids
        self.scache_feat = self.dataset.scache_feat
        self.scache_size = len(self.scache_nids)

        part_sizes = [len(part) for part in self.parts]
        part_sizes = torch.LongTensor([0] + part_sizes)
        self.nids_pptr = torch.cumsum(part_sizes, dim=0)

        # pin some data in memory
        train_idx = self.dataset.get_idx_split()['train']
        train_mask = torch.zeros(gnnos_dataset.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        self.train_mask = train_mask
        self.labels = self.dataset.labels.clone()
        self.pg_pptr = self.pg.part_ptr.clone()    # partition ptrs
        self.scache_pptr = self.scache.part_ptr.clone()
        # s-node induced graph
        start, end = self.scache_pptr[self.psize], self.scache_pptr[self.psize+1]
        scache_srcs = self.scache.src_nids[start:end].clone()
        scache_dsts = self.scache.dst_nids[start:end].clone()
        dtype = self.scache_nids.numpy().dtype
        scache_dict = gp.Dict(dtype, dtype, default_value=-1)
        scache_dict[self.scache_nids.numpy()] = torch.arange(self.scache_size).numpy()
        self.scache_srcs = torch.from_numpy(scache_dict[scache_srcs])
        self.scache_dsts = torch.from_numpy(scache_dict[scache_dsts])
        assert (self.scache_srcs != -1).all()
        assert (self.scache_dsts != -1).all()

        self.n = 0
        self.max = int((self.psize) // self.bsize)
        self.train_pids = torch.randperm(self.psize)

        # share common data
        for p in self.parts:
            p.share_memory_()
        self.scache_nids.share_memory_()
        self.scache_feat.share_memory_()
        self.nids_pptr.share_memory_()
        self.train_mask.share_memory_()
        self.labels.share_memory_()
        self.pg_pptr.share_memory_()
        self.scache_pptr.share_memory_()
        self.scache_srcs.share_memory_()
        self.scache_dsts.share_memory_()

        # spawn workers for data loading and filtering
        context = mp.get_context("fork")
        self.prep_q = context.Queue(maxsize=1)
        self.post_q = context.Queue(maxsize=1)
        # self.index_q = context.Queue(maxsize=1)
        self.loader = context.Process(target=self.load_store, args=())
        self.assembler = context.Process(target=self.assemble, args=())

    def assemble(self):
        while True:
            pids, batch_feat, (batch_srcs, batch_dsts), (scache_batch_srcs, scache_batch_dsts) = self.prep_q.get()
            tic = time.time()
            scache_nids, scache_size = self.scache_nids, self.scache_size
            dcache_nids = torch.cat([self.parts[i] for i in pids])
            dcache_size = len(dcache_nids)
            cache_size = scache_size + dcache_size
            scache_nids_m = torch.arange(self.scache_size).numpy()
            batch_nids_m = torch.arange(self.scache_size, cache_size).numpy()
            dtype = scache_nids_m.dtype
            gp_dict = gp.Dict(dtype, dtype, default_value=-1)
            # insert new id int dict
            gp_dict[dcache_nids.numpy()] = batch_nids_m
            gp_dict[scache_nids.numpy()] = scache_nids_m

            # processing after data is loaded
            scache_batch_srcs = lookup(gp_dict, scache_batch_srcs)
            scache_batch_dsts = lookup(gp_dict, scache_batch_dsts)
            batch_dsts = lookup(gp_dict, batch_dsts)
            mask = (batch_dsts == -1)
            batch_srcs, batch_dsts = batch_srcs[mask], batch_dsts[mask]
            batch_srcs = lookup(gp_dict, batch_srcs)

            batch_srcs = torch.cat((self.scache_srcs, scache_batch_srcs, batch_srcs))
            batch_dsts = torch.cat((self.scache_dsts, scache_batch_dsts, batch_dsts))
            batch_labels = torch.cat((self.labels[scache_nids], self.labels[dcache_nids]))
            batch_feat = torch.cat((self.scache_feat, batch_feat))
            batch_train_mask = torch.zeros((cache_size,), dtype=torch.bool)
            batch_train_mask[scache_size:] = self.train_mask[dcache_nids]
            toc = time.time()
            print(f"assemble: {toc-tic:.2f}s")

            self.post_q.put((cache_size, (batch_srcs, batch_dsts), batch_labels))
        
    def load_store(self):
        while True:
            tic = time.time()
            print("start loading")
            sample_pids = self.train_pids[self.n*self.bsize : (self.n+1)*self.bsize]

            ranges = []
            for i in sample_pids:
                ranges.append((self.nids_pptr[i], self.nids_pptr[i+1]))
            print("load feats")
            print(self.dataset.node_feat)
            print(self.dataset.node_feat.is_shared())
            batch_feat = gnnos.gather_tensor_slices(self.dataset.node_feat, ranges)

            ranges = []
            for i in sample_pids:
                ranges.append((self.pg_pptr[i], self.pg_pptr[i+1]))
            print("load partitions")
            batch_srcs = gnnos.gather_tensor_slices(self.pg.src_nids, ranges)
            batch_dsts = gnnos.gather_tensor_slices(self.pg.dst_nids, ranges)
            print(f"raw edges: {len(batch_srcs)}")

            ranges = []
            for i in sample_pids:
                ranges.append((self.scache_pptr[i], self.scache_pptr[i+1]))
            scache_batch_srcs = gnnos.gather_tensor_slices(self.scache.src_nids, ranges)
            scache_batch_dsts = gnnos.gather_tensor_slices(self.scache.dst_nids, ranges)
            toc = time.time()
            print(f"Load store: {toc-tic:.2f}s")

            self.prep_q.put(
                (sample_pids, batch_feat, (batch_srcs, batch_dsts), (scache_batch_srcs, scache_batch_dsts))
            )
            # self.index_q.put(self.n)
            self.n += 1
            if self.n == self.max:
                self.train_pids = torch.randperm(self.psize)
                # self.index_q.put(self.max)
                self.n = 0

    def start(self):
        self.loader.start()
        # self.assembler.start()

    def join(self):
        self.loader.join()
        self.assembler.join()

    # def __len__(self):
    #     return self.max

    # def __next__(self):
    #     index = self.index_q.get()
    #     if index < self.max:
    #         return self.post_q.get()
    #     else:
    #         raise StopIteration

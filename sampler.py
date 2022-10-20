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
        # self.train_pids = torch.randperm(self.psize)
        self.train_pids = torch.arange(self.psize)

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
            return subgraph, sg_train_nids, # train_nids, helper_nids, cache_nids
        else:
            self.train_pids = torch.randperm(self.psize)
            self.node_cache_parts = self.__get_popular_nodes__()
            raise StopIteration

import gnnos
from graphloader import GnnosNodePropPredDataset
import torch.multiprocessing as mp

def new_dict(keys: torch.Tensor, values: torch.Tensor, default=-1):
    keys_np, values_np = keys.numpy(), values.numpy()
    gp_dict = gp.Dict(keys_np.dtype, values_np.dtype, default_value=default)
    gp_dict[keys_np] = values_np
    return gp_dict

def lookup(gp_dict: gp.Dict, tensor: torch.Tensor):
    return torch.from_numpy(gp_dict[tensor.numpy()])

class GnnosIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis/other partitioners is used as the graph partition backend.
    The sampler returns a subgraph induced by a batch of clusters
    '''
    def __init__(self, gnnos_dataset: GnnosNodePropPredDataset, bsize,
        share_memory=False, use_old_feat=False):
        self.share_memory = share_memory
        self.use_old_feat = use_old_feat

        self.dataset = gnnos_dataset
        self.psize = gnnos_dataset.psize
        self.bsize = bsize
        # train_mask and assigns are in-memory tensors - O(|V|)
        self.dcache = dict(zip(['graph', 'assigns', 'parts', 'feat', 'label'], [
            gnnos_dataset.graph, gnnos_dataset.assigns, gnnos_dataset.parts,
            gnnos_dataset.node_feat, gnnos_dataset.labels
        ]))
        part_sizes = [len(part) for part in self.dcache['parts']]
        part_sizes = torch.LongTensor([0] + part_sizes)
        self.dcache['node_ptr'] = torch.cumsum(part_sizes, dim=0)
        self.dcache['edge_ptr'] = self.dcache['graph'].part_ptr.tensor()
        assert self.dcache['edge_ptr'].shape[0] == self.psize + 1
        train_idx = self.dataset.get_idx_split()['train']
        train_mask = torch.zeros(gnnos_dataset.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        self.dcache['train_mask'] = train_mask

        self.scache = dict(zip(['graph', 'nids', 'feat', 'label'], self.dataset.scache))
        # get a partitioning of scache_nids
        self.scache['assigns'] = self.dcache['assigns'][self.scache['nids']]
        self.scache['parts'] = partition_from(self.scache['nids'], self.scache['assigns'],
            self.psize)
        self.scache['edge_ptr'] = self.scache['graph'].part_ptr.tensor()
        assert self.scache['edge_ptr'].shape[0] == self.psize + 2
        self.scache['sg'] = self.scache['graph'][self.psize]

        print("Relabeling scache induced subgraph")
        tic = time.time()
        self.scache_mapping = new_dict(
            self.scache['nids'], torch.arange(len(self.scache['nids'])), default=-1)
        self.scache['sg'][0] = lookup(self.scache_mapping, self.scache['sg'][0])
        self.scache['sg'][1] = lookup(self.scache_mapping, self.scache['sg'][1])
        assert (self.scache['sg'] != -1).all()
        print(f"Done: {time.time()-tic:.2f}s")

        self.max = int((self.psize) // self.bsize)
        # self.train_pids = torch.randperm(self.psize)
        self.train_pids = torch.arange(self.psize)
        self.relabel_dict = torch.empty(self.dataset.num_nodes, dtype=torch.long)  # at most ~2GB

    def load_store(self, pi):
        tic = time.time()
        sample_pids = self.train_pids[pi*self.bsize : (pi+1)*self.bsize]
        if self.use_old_feat:
            dcache_nids = torch.cat([self.dcache['parts'][i] for i in sample_pids])
            # different results, same computation
            batch_feat = self.dcache['feat'][dcache_nids]
        else:
            ranges = []
            for i in sample_pids:
                ranges.append((self.dcache['node_ptr'][i], self.dcache['node_ptr'][i+1]))
            batch_feat = self.dcache['feat'].gather(ranges)
            batch_labels = self.dcache['label'].gather(ranges)
            range_sizes = [r[1]-r[0] for r in ranges]
            # print(f"slice size avg={np.mean(range_sizes):.2f}, std={np.std(range_sizes):.2f}")

        ranges = []
        for i in sample_pids:
            ranges.append((self.dcache['edge_ptr'][i], self.dcache['edge_ptr'][i+1]))
        range_sizes = [r[1]-r[0] for r in ranges]
        # print(f"coo slice size avg={np.mean(range_sizes):.2f}, std={np.std(range_sizes):.2f}")
        batch_srcs = self.dcache['graph'].src_nids.gather(ranges)
        batch_dsts = self.dcache['graph'].dst_nids.gather(ranges)
        print(f"parts edges: {len(batch_srcs)}")

        ranges = []
        for i in sample_pids:
            ranges.append((self.scache['edge_ptr'][i], self.scache['edge_ptr'][i+1]))
        scache_batch_srcs = self.scache['graph'].src_nids.gather(ranges)
        scache_batch_dsts = self.scache['graph'].dst_nids.gather(ranges)
        toc = time.time()
        print(f"Load store: {toc-tic:.2f}s")

        # processing after data is loaded
        tic = time.time()

        # prepare relabel dict
        scache_nids, scache_size = self.scache['nids'], len(self.scache['nids'])
        dcache_nids = torch.cat([self.dcache['parts'][i] for i in sample_pids])
        dcache_size = len(dcache_nids)
        # we are overestimating cache_size because some scache nids are in dcache nids
        # but it's fine
        cache_size = scache_size + dcache_size

        self.relabel_dict[:] = -1
        self.relabel_dict[scache_nids] = torch.arange(scache_size)
        self.relabel_dict[dcache_nids] = torch.arange(scache_size, cache_size)
        batch_srcs = self.relabel_dict[batch_srcs]
        # filter out non-cache destinations
        batch_dsts = self.relabel_dict[batch_dsts]
        dst_mask = (batch_dsts != -1)
        batch_srcs, batch_dsts = batch_srcs[dst_mask], batch_dsts[dst_mask]
        assert (batch_srcs != -1).all()
        assert (batch_dsts != -1).all()
        print(f"dcache edges: {len(batch_srcs)}")

        # NOTE: we are including slightly more edges than necessary - adjacency lists
        # of nodes that appear in both scache and dcache, but they won't appear in mfg
        scache_batch_dsts = self.relabel_dict[scache_batch_dsts]
        self.relabel_dict[scache_nids] = torch.arange(scache_size)
        scache_batch_srcs = self.relabel_dict[scache_batch_srcs]
        scache_dcache_nids = torch.cat([self.scache['parts'][i] for i in sample_pids])
        scache_dcache_nids = lookup(self.scache_mapping, scache_dcache_nids)
        scache_mask = torch.ones(scache_size, dtype=torch.bool)
        scache_mask[scache_dcache_nids] = False
        mask = scache_mask[scache_batch_srcs]
        scache_batch_srcs, scache_batch_dsts = scache_batch_srcs[mask], scache_batch_dsts[mask]
        scache_sg_srcs, scache_sg_dsts = self.scache['sg']
        mask = scache_mask[scache_sg_srcs]
        scache_sg = scache_sg_srcs[mask], scache_sg_dsts[mask]
        print(f"scache edges: {len(scache_batch_srcs)}, {len(scache_sg[0])}")

        # assemble train_mask
        batch_train_mask = torch.zeros((cache_size,), dtype=torch.bool)
        batch_train_mask[scache_size:cache_size] = self.dcache['train_mask'][dcache_nids]

        batch_srcs = torch.cat((scache_sg[0], scache_batch_srcs, batch_srcs))
        batch_dsts = torch.cat((scache_sg[1], scache_batch_dsts, batch_dsts))
        batch_labels = torch.cat((self.scache['label'], batch_labels))
        batch_feat = torch.cat((self.scache['feat'], batch_feat))

        # if self.share_memory:
        #     batch_srcs.share_memory_()
        #     batch_dsts.share_memoty_()
        #     batch_labels.share_memory_()
        #     batch_feat.share_memory_()
        #     batch_train_mask.share_memory_()
        toc = time.time()
        print(f"Assemble: {toc-tic:.2f}s")
        return cache_size, (batch_srcs, batch_dsts), batch_labels, batch_feat, batch_train_mask

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            print("Iteration", self.n)
            self.n += 1
            return self.load_store(self.n-1)
        else:
            self.n = 0
            self.train_pids = torch.randperm(self.psize)
            raise StopIteration

from graphloader import BaselineNodePropPredDataset

if __name__ == "__main__":
    import argparse
    from pyinstrument import Profiler
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-papers100M')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--psize", type=int, default=16384)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--io-threads", type=int, default=32,
                        help="threads to load data from storage (could be larger than #cpus)")
    args = parser.parse_args()

    print(args)
    print("set NUM_IO_THREADS to", args.io_threads)
    gnnos.set_io_threads(args.io_threads)
    # torch.set_num_threads(args.io_threads)

    baseline_data = BaselineNodePropPredDataset(name=args.dataset, root=args.root, mmap_feat=True)
    g = baseline_data.graph
    g.ndata['label'] = baseline_data.labels
    n_nodes = g.num_nodes()
    idx = baseline_data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    g.ndata['train_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['train_mask'][train_nid] = True
    g.ndata['valid_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['valid_mask'][val_nid] = True
    g.ndata['test_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['test_mask'][test_nid] = True
    baseline_it = iter(ClusterIterV2(args.dataset, g, args.psize, args.bsize, args.bsize,
        partitioner=partition_utils.MetisMinCutBalanced(), popular_ratio=0.01))

    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize, topk=0.01)
    it = iter(GnnosIter(data, args.bsize))

    iters = 0
    duration = []
    for i in range(args.n_epochs):
        print("Loading starts")
        for _ in range(len(it)):
            baseline_data = next(baseline_it)
            data = next(it)
            num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask = data
            print("baseline graph:", baseline_data[0])
            print(f"gnnos nodes:{num_nodes} edges:{len(batch_coo[0])}")
            assert baseline_data[0].num_edges() == len(batch_coo[0])

        tic = time.time()
        # profiler = Profiler(interval=0.01)
        # profiler.start()
        # for data in it:
        #     num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask = data
        #     print(f"#nodes: {num_nodes}, #edges: {len(batch_coo[0])}, #train: {batch_train_mask.int().sum()}")
        #     print(f"batch_feat: {batch_feat.shape}")
        #     assert num_nodes == batch_feat.shape[0]
        #     assert num_nodes == batch_labels.shape[0]
        #     assert num_nodes == batch_train_mask.shape[0]
        #     graph = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
        #     graph.create_formats_()
        #     del num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask
        #     profiler.stop()
        #     profiler.print()
        #     profiler.start()
        # profiler.stop()
        toc = time.time()
        print(f"{len(it)} iters took {toc-tic:.2f}s")
        duration.append(toc-tic)

    print(f"On average: {np.mean(duration):.2f}")

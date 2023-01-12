import os, time
from tokenize import Double
import numpy as np
import torch, dgl
import torch.multiprocessing as mp
import dgl.function as fn

import partition_utils
from partition_utils import partition_from

def identify_cut_edges(g, assigns):
    assert g.num_nodes() == assigns.shape[0]
    with g.local_scope():
        g.ndata['vp'] = assigns.float()
        g.apply_edges(fn.u_sub_v('vp', 'vp', 'cut'))
        return g.edata['cut'] != 0

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

class ClusterIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis/other partitioners is used as the graph partition backend.
    The sampler returns a subgraph induced by a batch of clusters
    '''
    def __init__(self, dataset, g, psize, bsize, hsize,
            partitioner=partition_utils.RandomNodePartitioner(), seed=1,
            sample_helpers=False, sample_topk=False, popular_ratio=0):
        self.g = g
        self.psize = psize
        self.bsize = bsize
        self.sample_topk = sample_topk
        self.sample_helpers = sample_helpers

        cache_folder = os.path.join(os.environ['DATASETS'], "gnnos",
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
        
        g.edata['cut'] = identify_cut_edges(g, self.assigns)
        nontrain_mask = ~g.ndata['train_mask']
        self.parts = partition_from(nids[nontrain_mask], self.assigns[nontrain_mask], psize)
        self.train_parts = partition_from(train_nids, self.assigns[train_nids], psize)
        self.train_nids = train_nids

        torch.manual_seed(seed)
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
            # NOTE: check cache_nids don't duplicate with other nodes
            assert len(torch.unique(nids)) == len(nids)

            subgraph = dgl.node_subgraph(self.g, nids)
            old_eids = subgraph.edata[dgl.EID]
            subgraph.edata['cut'] = self.g.edata['cut'][old_eids]
            sg_num_nodes = subgraph.num_nodes()
            sg_train_nids = torch.arange(train_nids.shape[0])
            sg_cache_nids = torch.arange(sg_num_nodes-len(cache_nids), sg_num_nodes)
            # NOTE: verify all sg_train_nids has train_mask
            assert (subgraph.ndata['train_mask'][sg_train_nids] == True).all().item()
            subgraph.ndata['train_mask'][:] = False
            subgraph.ndata['train_mask'][sg_train_nids] = True
            subgraph.ndata['cache_mask'] = torch.BoolTensor(sg_num_nodes)
            subgraph.ndata['cache_mask'][:] = False
            subgraph.ndata['cache_mask'][sg_cache_nids] = True
            self.n += 1
            return subgraph, sg_train_nids, train_nids, helper_nids, cache_nids
        else:
            self.train_pids = torch.randperm(self.psize)
            self.node_cache_parts = self.__get_popular_nodes__()
            raise StopIteration

import gnnos
from graphloader import GnnosNodePropPredDataset
#  from memory_profiler import profile

class GnnosIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis/other partitioners is used as the graph partition backend.
    The sampler returns a subgraph induced by a batch of clusters
    '''
    def __init__(self, gnnos_dataset: GnnosNodePropPredDataset, bsize,
        use_old_feat=False, seed=1):
        self.use_old_feat = use_old_feat

        self.dataset = gnnos_dataset
        self.psize = gnnos_dataset.psize
        self.bsize = bsize
        # train_mask and assigns are in-memory tensors - O(|V|)
        self.dcache = dict(zip(['graph', 'parts', 'feat', 'label'], [
            gnnos_dataset.graph, gnnos_dataset.parts,
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
        scache_assigns = gnnos_dataset.assigns[self.scache['nids']]
        self.scache['parts'] = partition_from(self.scache['nids'], scache_assigns, self.psize)
        # del gnnos_dataset.assigns
        self.scache['edge_ptr'] = self.scache['graph'].part_ptr.tensor()
        assert self.scache['edge_ptr'].shape[0] == self.psize + 2
        self.scache['sg'] = self.scache['graph'][self.psize]

        torch.manual_seed(seed)
        self.max = int((self.psize) // self.bsize)
        self.train_pids = torch.randperm(self.psize)

    #  @profile
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
        scache_batch_srcs = torch.cat((scache_batch_srcs, self.scache['sg'][0]))
        scache_batch_dsts = torch.cat((scache_batch_dsts, self.scache['sg'][1]))
        toc = time.time()
        print(f"Load store: {toc-tic:.2f}s")

        # prepare relabel dict
        scache_nids, scache_size = self.scache['nids'], len(self.scache['nids'])
        dcache_nids = torch.cat([self.dcache['parts'][i] for i in sample_pids])
        dcache_size = len(dcache_nids)
        # we are overestimating cache_size because some scache nids are in dcache nids
        # but it's fine
        cache_size = scache_size + dcache_size
        # nodes that are in both scache and dcache
        scache_dcache_nids = torch.cat([self.scache['parts'][i] for i in sample_pids])

        relabel_dict = torch.empty(self.dataset.num_nodes, dtype=torch.long)  # at most ~2GB
        relabel_dict[:] = -1
        relabel_dict[scache_nids] = torch.arange(scache_size)
        relabel_dict[dcache_nids] = torch.arange(scache_size, cache_size)
        #  batch_srcs = relabel_dict[batch_srcs]
        batch_srcs = torch.index_select(relabel_dict, 0, batch_srcs)
        # filter out non-cache destinations
        #  batch_dsts = relabel_dict[batch_dsts]
        batch_dsts = torch.index_select(relabel_dict, 0, batch_dsts)
        #  print(f"relabel dcache: {time.time()-tic:.2f}s")
        edge_mask = (batch_dsts != -1)
        batch_srcs, batch_dsts = batch_srcs[edge_mask], batch_dsts[edge_mask]
        #  print(f"filter dcache: {time.time()-tic:.2f}s")
        assert (batch_srcs != -1).all()
        assert (batch_dsts != -1).all()
        print(f"dcache edges: {len(batch_srcs)}, {time.time()-tic:.2f}s")

        # deduplicate adj lists of nodes that appears in both scache and dcache
        scache_batch_dsts = relabel_dict[scache_batch_dsts]
        relabel_dict[scache_dcache_nids] = -1
        scache_batch_srcs = relabel_dict[scache_batch_srcs]
        #  print(f"relabel scache: {time.time()-tic:.2f}s")
        del relabel_dict
        edge_mask = (scache_batch_srcs != -1)
        scache_batch_srcs, scache_batch_dsts = scache_batch_srcs[edge_mask], scache_batch_dsts[edge_mask]
        #  print(f"filter scache: {time.time()-tic:.2f}s")
        print(f"scache edges: {len(scache_batch_srcs)}, {time.time()-tic:.2f}s")

        # assemble train_mask
        batch_train_mask = torch.zeros((cache_size,), dtype=torch.bool)
        batch_train_mask[scache_size:cache_size] = self.dcache['train_mask'][dcache_nids]

        batch_srcs = torch.cat((scache_batch_srcs, batch_srcs))
        batch_dsts = torch.cat((scache_batch_dsts, batch_dsts))
        batch_labels = torch.cat((self.scache['label'], batch_labels))
        batch_feat = torch.cat((self.scache['feat'], batch_feat))

        toc = time.time()
        print(f"Assemble: {toc-tic:.2f}s")
        return cache_size, (batch_srcs, batch_dsts), batch_labels, batch_feat, batch_train_mask, \
            scache_nids, dcache_nids

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

# make sure the megabatch graphs from in-memory ClusterIterV2 and out-of-core GnnosIter match
if __name__ == "__main__":
    import argparse
    #  from pyinstrument import Profiler
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-papers100M')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--psize", type=int, default=16384)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--io-threads", type=int, default=32,
                        help="threads to load data from storage (could be larger than #cpus)")
    parser.add_argument("--seed", type=int, default=1,
                        help="common seed to make sure loaders share the same beginning state")
    args = parser.parse_args()

    print(args)
    print("set NUM_IO_THREADS to", args.io_threads)
    gnnos.set_io_threads(args.io_threads)
    # torch.set_num_threads(args.io_threads)
    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize, topk=0.01)
    it = iter(GnnosIter(data, args.bsize, seed=args.seed))

    for _ in range(args.n_epochs):
        for data in it:
            del data

    #  ctx = mp.get_context('spawn')
    #  data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize, topk=0.01)
    #  it = iter(GnnosIterShm(data, args.bsize, ctx=ctx))
    #  loader = ctx.Process(target=compare_partition, args=(it, args))
    #  loader.start()
    #  loader.join()

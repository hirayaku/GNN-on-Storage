import os, time
import torch, dgl
import torch.multiprocessing as mp

import gnnos
import sampler, partition_utils
from graphloader import GnnosNodePropPredDataset

class DoubleBuffer(object):
    def __init__(self, ctx, shape, dtype):
        self.buf = torch.empty(shape, dtype=dtype)
        self.cap = shape[0]
        self.buf_free = ctx.Array('i', (0, self.cap))
        self.buf.share_memory_()

        self.sem_full = ctx.Semaphore(0)
        self.sem_free = ctx.Semaphore(2)
        self.alloc_lock = ctx.Lock()

    def __getitem__(self, s):
        return self.buf[s]

    def reserve(self, num_slots):
        self.sem_free.acquire()
        with self.alloc_lock:
            if self.buf_free[1] - self.buf_free[0] < num_slots:
                self.sem_free.release()
                raise RuntimeError(f"Insufficient edge buffer")
            if self.buf_free[0] == 0:
                self.buf_free[0] = num_slots
                interval = 0, num_slots
            else:
                self.buf_free[1] = self.buf_free[1] - num_slots
                interval = self.buf_free[1], self.buf_free[1] + num_slots
            # print("Buffer reserved:", interval)
            self.sem_full.release()
            return interval

    def free(self, interval):
        self.sem_full.acquire()
        buf_slots = None
        with self.alloc_lock:
            if interval[1] == self.buf_free[0]:
                self.buf_free[0] = interval[0]
            elif interval[0] == self.buf_free[1]:
                self.buf_free[1] = interval[1]
            else:
                self.sem_full.release()
                raise RuntimeError(f"invalid buffer state")
            buf_slots = self.buf_free[:]
        # print("Buffer free:", buf_slots)
        self.sem_free.release()

# from memory_profiler import profile
class GnnosIterShm(sampler.GnnosIter):
    '''
    Directly constructs the megabatch in the shared memory

    '''
    # @profile
    def __init__(self, gnnos_dataset: GnnosNodePropPredDataset, bsize: int, seed=1, ctx=None):
        super(GnnosIterShm, self).__init__(gnnos_dataset, bsize, seed=seed)
        self.ctx = ctx or mp.get_context('spawn')
        # pre-allocate some buffers to use later
        npart_sizes = self.dcache['node_ptr'][1:] - self.dcache['node_ptr'][:-1]
        scache_size = len(self.scache['nids'])
        epart_sizes = self.dcache['edge_ptr'][1:] - self.dcache['edge_ptr'][:-1]
        spart_sizes = self.scache['edge_ptr'][1:] - self.scache['edge_ptr'][:-1]
        # TODO: we are wasting memory for 2*scache_size nodes; we don't have to keep copies of them
        node_buf_size = torch.topk(npart_sizes, 2*bsize).values.sum() + 2 * scache_size
        edge_buf_size = 2*int(epart_sizes.float().mean().item() * bsize) + 2*len(self.scache['sg'][0])
        # edge_buf_size = 2 * (int(epart_sizes.float().mean().item() * bsize) +
        #     int(spart_sizes.float().mean().item() * bsize)) + 2 * len(self.scache['sg'][0])

        feat_dtype = self.dcache['feat'].metadata.dtype
        print(f"Buffer sizes: node[{feat_dtype},{node_buf_size}], edge[{edge_buf_size}]")
        self.src_buffer = DoubleBuffer(self.ctx, (edge_buf_size,), dtype=torch.long)
        self.dst_buffer = DoubleBuffer(self.ctx, (edge_buf_size,), dtype=torch.long)
        self.feat_buffer = DoubleBuffer(self.ctx, (node_buf_size, *self.dcache['feat'].shape[1:]),
            dtype=self.dcache['feat'].metadata.dtype)
        self.label_buffer = DoubleBuffer(self.ctx, (node_buf_size, *self.dcache['label'].shape[1:]),
            dtype=self.dcache['label'].metadata.dtype)
        self.data_queue = ctx.SimpleQueue()

    # @profile
    def load_store(self, pi):
        tic_0 = time.time()
        sample_pids = self.train_pids[pi*self.bsize : (pi+1)*self.bsize]

        # load node data
        ranges = []
        for i in sample_pids:
            ranges.append((self.dcache['node_ptr'][i], self.dcache['node_ptr'][i+1]))
        dcache_slots = sum([r[1]-r[0] for r in ranges]).item()
        scache_slots = self.scache['feat'].shape[0]
        # could block here
        feat_interval = self.feat_buffer.reserve(scache_slots + dcache_slots)
        label_interval = self.label_buffer.reserve(scache_slots + dcache_slots)
        tic = time.time()
        # TODO: add a version of TensorStore.gather_to
        self.feat_buffer[slice(*feat_interval)][:scache_slots] = self.scache['feat']
        self.feat_buffer[slice(*feat_interval)][scache_slots:] = self.dcache['feat'].gather(ranges)
        self.label_buffer[slice(*label_interval)][:scache_slots] = self.scache['label']
        self.label_buffer[slice(*label_interval)][scache_slots:] = self.dcache['label'].gather(ranges)

        # load and process edge data
        ranges = []
        for i in sample_pids:
            ranges.append((self.dcache['edge_ptr'][i], self.dcache['edge_ptr'][i+1]))
        batch_srcs = self.dcache['graph'].src_nids.gather(ranges)
        batch_dsts = self.dcache['graph'].dst_nids.gather(ranges)
        print(f"[L] parts edges: {len(batch_srcs)}")

        ranges = []
        for i in sample_pids:
            ranges.append((self.scache['edge_ptr'][i], self.scache['edge_ptr'][i+1]))
        scache_batch_srcs = self.scache['graph'].src_nids.gather(ranges)
        scache_batch_dsts = self.scache['graph'].dst_nids.gather(ranges)
        scache_batch_srcs = torch.cat((scache_batch_srcs, self.scache['sg'][0]))
        scache_batch_dsts = torch.cat((scache_batch_dsts, self.scache['sg'][1]))
        toc = time.time()
        print(f"[L] Buf reserve: {tic-tic_0:.2f}s")
        print(f"[L] Load store:  {toc-tic:.2f}s")

        # prepare relabel dict
        scache_nids, scache_size = self.scache['nids'], len(self.scache['nids'])
        dcache_nids = torch.cat([self.dcache['parts'][i] for i in sample_pids])
        dcache_size = len(dcache_nids)
        cache_size = scache_size + dcache_size
        # nodes that are in both scache and dcache
        scache_dcache_nids = torch.cat([self.scache['parts'][i] for i in sample_pids])

        relabel_dict = torch.empty(self.dataset.num_nodes, dtype=torch.long)  # at most ~2GB
        relabel_dict[:] = -1
        relabel_dict[scache_nids] = torch.arange(scache_size)
        relabel_dict[dcache_nids] = torch.arange(scache_size, cache_size)
        # filter out non-cache destinations
        batch_srcs = relabel_dict[batch_srcs]
        batch_dsts = relabel_dict[batch_dsts]
        print(f"[L] Relabel dcache: {time.time()-tic:.2f}s")
        dcache_edge_mask = (batch_dsts != -1)

        # deduplicate adj lists of nodes that appears in both scache and dcache
        scache_batch_dsts = relabel_dict[scache_batch_dsts]
        relabel_dict[scache_dcache_nids] = -1
        scache_batch_srcs = relabel_dict[scache_batch_srcs]
        del relabel_dict
        print(f"[L] Relabel scache: {time.time()-tic:.2f}s")
        scache_edge_mask = (scache_batch_srcs != -1)

        scache_slots = scache_edge_mask.int().sum().item()
        dcache_slots = dcache_edge_mask.int().sum().item()
        src_interval = self.src_buffer.reserve(scache_slots + dcache_slots)
        dst_interval = self.dst_buffer.reserve(scache_slots + dcache_slots)
        self.src_buffer[slice(*src_interval)][:scache_slots] = scache_batch_srcs[scache_edge_mask]
        self.src_buffer[slice(*src_interval)][scache_slots:] = batch_srcs[dcache_edge_mask]
        del scache_batch_srcs, batch_srcs
        self.dst_buffer[slice(*dst_interval)][:scache_slots] = scache_batch_dsts[scache_edge_mask]
        self.dst_buffer[slice(*dst_interval)][scache_slots:] = batch_dsts[dcache_edge_mask]
        del scache_batch_dsts, batch_dsts

        # assemble train_mask
        batch_train_mask = torch.zeros((cache_size,), dtype=torch.bool)
        batch_train_mask[scache_size:cache_size] = self.dcache['train_mask'][dcache_nids]

        toc = time.time()
        print(f"[L] Assemble: {toc-tic:.2f}s")
        # data = (cache_size,
        #     src_interval, dst_interval, label_interval, feat_interval,
        #     batch_train_mask, scache_nids, dcache_nids)
        data = cache_size, src_interval, dst_interval, label_interval, feat_interval, batch_train_mask
        self.data_queue.put(('train', data))
        return data 
    
    def evaluate(self):
        pass
    
    def finish(self):
        self.data_queue.put('end')

from graphloader import BaselineNodePropPredDataset

def retrieve_data(data_queue: mp.SimpleQueue, buffers, args):
    print("[W] Worker starts")
    src_buffer, dst_buffer, label_buffer, feat_buffer = buffers
    for _ in range(args.n_epochs):
        while True:
            tic = time.time()
            data = data_queue.get()
            if data[0] != 'train':
                break
            num_nodes, src_interval, dst_interval, label_interval, feat_interval, \
                batch_train_mask = data[1]
            batch_coo = (src_buffer[slice(*src_interval)], dst_buffer[slice(*dst_interval)])
            batch_labels = label_buffer[slice(*label_interval)]
            batch_feat = feat_buffer[slice(*feat_interval)]
            print(f"[W] retrive from queue: {time.time()-tic:.2f}s")

            sg = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
            sg.create_formats_()
            print(f"[W] dgl graph creation: {time.time()-tic:.2f}s")
            print(f"[W] gnnos graph: ({sg.num_nodes()}, {sg.num_edges()})")

            src_buffer.free(src_interval)
            dst_buffer.free(dst_interval)
            label_buffer.free(label_interval)
            feat_buffer.free(feat_interval)

def compare_partition(data_queue: mp.SimpleQueue, buffers, args):
    baseline_data = BaselineNodePropPredDataset(name=args.dataset, root=args.root, mmap_feat=False)
    g = baseline_data.graph
    g.ndata['label'] = baseline_data.labels
    g.ndata['feat'] = baseline_data.node_feat
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
    baseline_it = iter(sampler.ClusterIterV2(args.dataset, g, args.psize, args.bsize, 0,
        partitioner=partition_utils.MetisMinCutBalanced(), popular_ratio=0.01))

    print("[W] Worker starts")
    src_buffer, dst_buffer, label_buffer, feat_buffer = buffers
    for _ in range(args.n_epochs):
        for _ in range(len(baseline_it)):
            tic = time.time()
            data = data_queue.get()
            if data[0] != 'train':
                break
            # scache_nids, dcache_nids
            num_nodes, src_interval, dst_interval, label_interval, feat_interval, \
                batch_train_mask, scache_nids, dcache_nids = data[1]
            batch_coo = (src_buffer[slice(*src_interval)], dst_buffer[slice(*dst_interval)])
            batch_labels = label_buffer[slice(*label_interval)]
            batch_feat = feat_buffer[slice(*feat_interval)]
            print(f"[W] retrive from queue: {time.time()-tic:.2f}s")

            sg = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
            sg.create_formats_()
            print(f"[W] dgl graph creation: {time.time()-tic:.2f}s")

            baseline_sg, baseline_train, *_ = next(baseline_it)
            print(f"[W] baseline graph: ({baseline_sg.num_nodes()}, {baseline_sg.num_edges()})")
            print(f"[W] gnnos graph: ({sg.num_nodes()}, {sg.num_edges()})")
            print(f"[W] baseline megabatch: {time.time()-tic:.2f}s")

            old_nids = torch.cat([scache_nids, dcache_nids])
            gnnos_train = batch_train_mask.nonzero(as_tuple=True)[0]
            assert baseline_sg.num_edges() == sg.num_edges()
            assert (g.ndata['feat'][old_nids] == batch_feat).all()
            # filter out NaN labels because NaN != NaN in Python
            ref_labels = g.ndata['label'][old_nids]
            ref_mask = ~torch.isnan(ref_labels)
            assert (torch.isnan(ref_labels) == torch.isnan(batch_labels)).all()
            assert (ref_labels[ref_mask] == batch_labels[ref_mask]).all()
            assert (g.ndata['train_mask'][dcache_nids] == batch_train_mask[len(scache_nids):]).all()
            assert len(baseline_train) == len(gnnos_train)
            baseline_train_nids = torch.sort(baseline_sg.ndata[dgl.NID][baseline_train])[0]
            gnnos_train_nids = torch.sort(old_nids[batch_train_mask])[0]
            assert (baseline_train_nids == gnnos_train_nids).all()
            assert (baseline_sg.in_degrees(baseline_train).sort()[0] == sg.in_degrees(gnnos_train).sort()[0]).all()
            print(f"[W] sanity check: {time.time()-tic:.2f}s")

            train_size = len(baseline_train)
            ns_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            baseline_dl = iter(dgl.dataloading.DataLoader(
                baseline_sg,
                baseline_train,
                ns_sampler,
                batch_size=train_size,
                shuffle=False,
                drop_last=False,
                num_workers=0))
            gnnos_dl = iter(dgl.dataloading.DataLoader(
                sg,
                batch_train_mask.nonzero(as_tuple=True)[0],
                ns_sampler,
                batch_size=train_size,
                shuffle=False,
                drop_last=False,
                num_workers=0))
            baseline_batch = next(baseline_dl)
            gnnos_batch = next(gnnos_dl)
            print("[W] baseline sampled blocks:\n", baseline_batch[-1])
            print("[W] gnnos sampled blocks:\n", gnnos_batch[-1])
            print(f"[W] neighbor sampling: {time.time()-tic:.2f}s")

            src_buffer.free(src_interval)
            dst_buffer.free(dst_interval)
            label_buffer.free(label_interval)
            feat_buffer.free(feat_interval)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-papers100M')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--psize", type=int, default=16384)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--io-threads", type=int, default=64,
                        help="threads to load data from storage (could be larger than #cpus)")
    parser.add_argument("--seed", type=int, default=1,
                        help="common seed to make sure loaders share the same beginning state")
    args = parser.parse_args()

    print(args)
    print("set NUM_IO_THREADS to", args.io_threads)
    gnnos.set_io_threads(args.io_threads)

    ctx = mp.get_context('spawn')
    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize, topk=0.01)
    loader = GnnosIterShm(data, bsize=args.bsize, ctx=ctx)
    del data
    shm_buffers = loader.src_buffer, loader.dst_buffer, loader.label_buffer, loader.feat_buffer
    worker = ctx.Process(target=retrieve_data, args=(loader.data_queue, shm_buffers, args))
    worker.start()

    for _ in range(args.n_epochs):
        tic = time.time()
        for data in loader:
            print("Gnnos loader:", data[1], data[2], data[3], data[4])
        print(f"Epoch time: {time.time()-tic:.2f}s")
        loader.evaluate()
    loader.finish()
    worker.join()

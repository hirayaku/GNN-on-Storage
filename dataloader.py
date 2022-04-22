import torch
import dgl
import dgl.dataloading
from graphloader import GraphLoader, PartitionedGraphLoader, PartitionSampler, partition_offsets

# Notes on dataloaders with multiprocessing
#
# 1. For pytorch dataloaders with num_workers > 0, when iterated (e.g. `for batch in dataloader:`)
# they return a multiprocessing-capable iterator through dataloader.__iter__().
# The mp-capable iterator spawns worker processes each with an index queue and a shared data queue.
# Each worker executes a loop getting indices, fetching data from the "dataset", invoking
# the provided `collate_fn` on the fetched data and put the result back in the data queue.
# `__next__` on the mp-capable iterator simply retries items from the data queue.
# The work done on the main process is indeed not much.
#
# 2. For dgl dataloaders, in order to perform graph-related sampling, DGL customizes
# the `collate_fn` passed to pytorch dataloader. `collate_fn` is a wrapper of
# the graph and the graph sampler. Graph sampling could then be done in parallel
# when num_workers > 0. However, there's one gotcha: feature slicing in DGLHeteroGraph
# is lazy. It's likely the feature of the sampled subgraph is not materialized until
# in the main process.

class _GNNoSIter(object):
    def __init__(self, it, graph_loader: GraphLoader):
        self.it = it
        self.gloader = graph_loader

    def __iter__(self):
        return self

    def __next__(self):
        input_nodes, output_nodes, blocks = next(self.it)
        # NB: not doing it inside dataloader workers to avoid data copy
        blocks[0].srcdata['feat'] = self.gloader.features(input_nodes)
        return input_nodes, output_nodes, blocks

class GNNoSDataLoader(dgl.dataloading.DataLoader):
    '''
    DataLoader with on-storage features; works with neighbor sampler
    '''
    def __init__(self, graph_loader, indices, sampler: dgl.dataloading.NeighborSampler,
                 device='cpu', use_ddp=False, ddp_seed=0,
                 batch_size=1, drop_last=False, shuffle=True, use_prefetch_thread=None,
                 use_alternate_stream=None, pin_prefetcher=None, use_uva=None, **kwargs):

        # TODO: when num_workers > 0, dgl DataLoader will invoke graph.create_formats_
        # it takes up too much memory
        super().__init__(
                graph_loader.graph, indices, sampler, device=device,
                use_ddp=use_ddp, ddp_seed=ddp_seed,
                batch_size=batch_size, drop_last=drop_last, shuffle=shuffle,
                use_prefetch_thread=use_prefetch_thread,
                use_alternate_streams=use_alternate_stream,
                pin_prefetcher=pin_prefetcher, use_uva=use_uva,
                **kwargs);

        self.gloader = graph_loader

    def __iter__(self):
        return _GNNoSIter(super().__iter__(), self.gloader)

class _GNNoSPartitionIter(object):
    def __init__(self, it, graph_loader: PartitionedGraphLoader):
        self.it = it
        self.gloader = graph_loader

    def __iter__(self):
        return self

    def __next__(self):
        sg, intervals, pids = next(self.it)
        # intervals = partition_offsets(sg_partitions)
        sg_features = torch.zeros((intervals[-1], *self.gloader.feature_dim()))
        for i in range(len(pids)):
            sg_features[intervals[i]:intervals[i+1]] = self.gloader.partition_features(pids[i])
        return sg, sg_features, intervals, pids 

class GNNoSPartitionDataLoader(dgl.dataloading.DataLoader):
    '''
    DataLoader with on-storage feature partitions
    '''
    #  TODO: spawn a separate processes to do the most work
    def __init__(self, graph_loader: PartitionedGraphLoader, batch_size, **kwargs):
        self.gloader = graph_loader
        super().__init__(
                self.gloader.graph, self.gloader.partition_idx(),
                PartitionSampler(self.gloader.partitions), device='cpu',
                batch_size=batch_size, drop_last=False, shuffle=True,
                **kwargs);

    def __iter__(self):
        return _GNNoSPartitionIter(super().__iter__(), self.gloader)

if __name__ == "__main__":
    from viztracer import VizTracer

    num_workers = 4
    num_partitions = 1200

    tracer = VizTracer(output_file=f"dataloader-w{num_workers}.json", min_duration=10)
    tracer.start()

    gloader = PartitionedGraphLoader(num_partitions, overwrite=False,
            name="ogbn-products", root="/mnt/md0/graphs", mmap=True)
    dataloader = GNNoSPartitionDataLoader(gloader, 20, num_workers=4)

    feature_dim = gloader.feature_dim()
    nodes = gloader.graph.nodes()

    for _, (sg, _) in enumerate(dataloader):
        assert (gloader.features(sg.ndata[dgl.NID]) == sg.ndata['feat']).all()

    del dataloader # to kill all worker processes; otherwise tracer won't stop

    tracer.stop()
    tracer.save()


from functools import partial
import psutil, torch
from data.graphloader import (
    NodePropPredDataset, ChunkedNodePropPredDataset,
    partition_dir, pivots_dir
)
from data.collater import Collator, CollatorPivots
from datapipe.custom_pipes import TensorShuffleWrapper, LiteIterableWrapper, ObjectAsPipe
from datapipe.custom_pipes import even_split_fn, shuffle_tensor
from datapipe.parallel_pipes import mp, make_dp_worker
from datapipe.sampler_fn import PygNeighborSampler, gather_feature, filter_and_pin

def make_shared(data):
    data.share_memory_()
    return data

class NodeDataLoader(object):
    def __init__(self, dataset: NodePropPredDataset, split: str, conf: dict, prefetch_cuda=True):
        self.ctx = mp.get_context('fork')
        self.cpu_i = 0

        datapipe = LiteIterableWrapper(dataset)
        index = dataset.get_idx_split()[split]
        index_dp = LiteIterableWrapper([index])
        if split == 'train':
            index_dp = index_dp.tensor_shuffle()
        index_dp = index_dp.map(make_shared)
        batch_size = conf['batch_size']
        datapipe = datapipe.zip(index_dp).map(list).flatmap(
            partial(even_split_fn, size=batch_size), flatten_col=1,
        )
        fanout = list(map(int, conf['fanout'].split(',')))
        sample_fn = PygNeighborSampler(fanout, filter_per_worker=False)
        num_par = conf.get('num_workers', 0)
        datapipe = datapipe.pmap(fn=sample_fn, mp_ctx=self.ctx, num_par=num_par)
        datapipe = datapipe.prefetch(
            fn=partial(gather_feature, filter_fn=filter_and_pin)
        )
        if prefetch_cuda:
            datapipe = datapipe.prefetch_cuda()
        self.cpu_i += num_par
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

    def shutdown(self):
        self.datapipe.reset()

from torch.utils.data import DataLoader

import queue
def relay(data, buf: queue.Queue):
    buf.put(data)
    return buf.get()

class NodeTorchDataLoader(object):
    def __init__(self, dataset: NodePropPredDataset, split: str, conf: dict):
        self.ctx = mp.get_context('fork')
        self.cpu_i = 0

        datapipe = LiteIterableWrapper(dataset)
        index_dp = LiteIterableWrapper([dataset.get_idx_split()[split]])
        if split == 'train':
            index_dp = index_dp.tensor_shuffle()
        index_dp = index_dp.map(make_shared)
        batch_size = conf['batch_size']
        datapipe = datapipe.zip(index_dp).map(fn=list).flatmap(
            # partial(split_fn, size=batch_size, drop_thres=1), flatten_col=1,
            partial(even_split_fn, size=batch_size), flatten_col=1,
        )
        # datapipe = make_dp_worker(datapipe, self.ctx, worker_name='index_feeder') # , num_par=4)
        # datapipe = datapipe.sharding_filter()

        fanout = list(map(int, conf['fanout'].split(',')))
        sample_fn = PygNeighborSampler(fanout, unbatch=True)
        num_par = conf.get('num_workers', 0)
        datapipe = DataLoader(
            datapipe, collate_fn=sample_fn,
            num_workers=num_par, persistent_workers=(num_par > 0)
        )
        self.cpu_i += num_par
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

    def shutdown(self):
        pass


def collate(collator: Collator, batch):
    return list(collator.collate(batch))

class PartitionDataLoader(object):
    def __init__(self, dataset_conf: dict, split: str, conf: dict):
        self.ctx = mp.get_context('fork')
        self.cpu_i = 0

        root = dataset_conf['root']
        method, P, batch_size = conf['partition'], conf['P'], conf['batch_size']
        datapipe = ObjectAsPipe(ChunkedNodePropPredDataset, partition_dir(root, method, P))
        if conf.get('pivots', False):
            pivot_dp = ObjectAsPipe(ChunkedNodePropPredDataset, pivots_dir(root, method, P),
                                    mmap={'graph': True, 'feat': False})
            datapipe = datapipe.zip(pivot_dp)
            datapipe = datapipe.map(partial(CollatorPivots, split=split), args=(0,1))
        else:
            datapipe = datapipe.map(partial(Collator, split=split))
        index_dp = TensorShuffleWrapper(torch.arange(P))
        datapipe = datapipe.zip(index_dp).map(list).flatmap(
            partial(even_split_fn, size=batch_size), flatten_col=1)
        num_repeats = conf.get('num_repeats', 1)
        datapipe = datapipe.map(collate, args=(0, 1)).repeats(num_repeats)
        if split == 'train':
            datapipe = datapipe.map(fn=shuffle_tensor, args=1, inplace=True)
        num_par = conf.get('num_workers', 0)
        if num_par > 0:
            datapipe = make_dp_worker(
                datapipe, self.ctx, worker_name='collate_worker', num_par=num_par,
                affinity=range(self.cpu_i, self.cpu_i+num_par))
        self.cpu_i += num_par

        self.method = method
        self.P = P
        self.batch_size = batch_size
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

    def __len__(self):
        return (self.P + self.batch_size - 1) // self.batch_size

    def shutdown(self):
        self.datapipe.reset()

class HierarchicalDataLoader(object):
    def __init__(self, dataset_conf: dict, split: str, conf: list[dict]):
        self.ctx = mp.get_context('fork')
        self.cpu_i = 0

        # Level-1 loader, partition-based
        partition_loader = PartitionDataLoader(dataset_conf, split, conf[0])
        datapipe = partition_loader.datapipe
        self.cpu_i += partition_loader.cpu_i
        # Level-2 loader, neighbor sampler
        conf_l2 = conf[1]
        batch_size = conf_l2['batch_size']
        fanout = list(map(int, conf_l2['fanout'].split(',')))
        sample_fn = PygNeighborSampler(fanout, filter_per_worker=False)
        num_par = conf_l2.get('num_workers',  0)
        pf_steps = conf_l2.get('prefetch_steps', max(100-num_par*2, 0))
        datapipe = datapipe.map(list).flatmap(
            partial(even_split_fn, size=batch_size), flatten_col=1,
        ).prefetch(buffer_size=pf_steps)
        datapipe = datapipe.pmap(fn=sample_fn, num_par=num_par, mp_ctx=self.ctx)
                                #  affinity=range(self.cpu_i, self.cpu_i+num_par))
        datapipe = datapipe.prefetch(
            fn=partial(gather_feature, filter_fn=filter_and_pin),
        ).prefetch_cuda()
        self.cpu_i += num_par

        self.batch_size = (partition_loader.batch_size, batch_size)
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

    def shutdown(self):
        self.datapipe.reset()

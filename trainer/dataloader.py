from functools import partial
import psutil, torch
from data.graphloader import (
    NodePropPredDataset, ChunkedNodePropPredDataset,
    partition_dir, pivots_dir
)
from data.collater import Collator, CollatorPivots
from datapipe.custom_pipes import TensorShuffleWrapper, LiteIterableWrapper, ObjectAsPipe 
from datapipe.custom_pipes import identity_fn, split_fn, even_split_fn
from datapipe.parallel_pipes import mp, make_dp_worker
from datapipe.sampler_fn import PygNeighborSampler

def get_idx_split(dataset: NodePropPredDataset, split: str):
    return dataset.get_idx_split(split)

def make_shared(data):
    data.share_memory_()
    return data

class NodeDataLoader(object):
    def __init__(self, dataset: dict, env: dict, split: str, conf: dict):
        self.ctx = mp.get_context('fork')
        self.ncpus = env.get('ncpus', psutil.cpu_count())
        self.cpu_i = 0

        ds = NodePropPredDataset(dataset['root'], mmap={'graph': True, 'feat': True}, 
                                 random=True, formats='csc')
        # ds[0].share_memory_()
        datapipe = LiteIterableWrapper(ds)
        index = ds.get_idx_split(split)
        index_dp = LiteIterableWrapper([index])
        if split == 'train':
            index_dp = index_dp.tensor_shuffle()
        index_dp = index_dp.map(make_shared)
        batch_size = conf['batch_size']
        drop_thres = 1 if conf.get('drop_last', True) else 0
        datapipe = datapipe.zip(index_dp).collate(collate_fn=list).flat_map(
            partial(split_fn, size=batch_size, drop_thres=drop_thres), flatten_col=1
        )
        fanout = list(map(int, conf['fanout'].split(',')))
        sample_fn = PygNeighborSampler(fanout)
        num_par = conf.get('num_workers', self.ncpus - self.cpu_i)
        datapipe = datapipe.par_map(fn=sample_fn, mp_ctx=self.ctx, num_par=num_par,
                                    affinity=range(self.cpu_i, self.cpu_i+num_par))

        self.cpu_i += num_par
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

from torch.utils.data import DataLoader

import queue
def relay(data, buf):
    buf.put(data)
    return buf.get()

class NodeTorchDataLoader(object):
    def __init__(self, dataset: dict, env: dict, split: str, conf: dict):
        self.ctx = mp.get_context('fork')
        self.ncpus = env.get('ncpus', psutil.cpu_count())
        self.cpu_i = 0

        ds = NodePropPredDataset(dataset['root'], mmap={'graph': True, 'feat': True}, 
                                 random=True, formats='csc')
        datapipe = LiteIterableWrapper(ds)
        index_dp = LiteIterableWrapper([ds.get_idx_split(split)])
        if split == 'train':
            index_dp = index_dp.tensor_shuffle()
        index_dp = index_dp.map(make_shared)
        batch_size = conf['batch_size']
        datapipe = datapipe.zip(index_dp).collate(collate_fn=list).flat_map(
            partial(split_fn, size=batch_size, drop_thres=1), flatten_col=1
        )
        # datapipe = make_dp_worker(datapipe, self.ctx, worker_name='index_feeder') # , num_par=4)
        datapipe = datapipe.sharding_filter()

        fanout = list(map(int, conf['fanout'].split(',')))
        sample_fn = PygNeighborSampler(fanout, unbatch=True)
        num_par = conf.get('num_workers', self.ncpus - self.cpu_i)
        datapipe = DataLoader(
            datapipe, collate_fn=sample_fn,
            num_workers=num_par, persistent_workers=(num_par > 0)
        )
        self.cpu_i += num_par
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)


def collate(collator: Collator, batch):
    return collator.collate(batch)

class PartitionDataLoader(object):
    def __init__(self, dataset: dict, env: dict, split: str, conf: dict):
        self.ctx = mp.get_context('fork')
        self.ncpus = env.get('ncpus', psutil.cpu_count())
        self.cpu_i = 0

        root = dataset['root']
        method, P, batch_size = conf['partition'], conf['P'], conf['batch_size']
        datapipe = ObjectAsPipe(ChunkedNodePropPredDataset, partition_dir(root, method, P))
        if conf.get('pivots', False):
            pivot_dp = ObjectAsPipe(ChunkedNodePropPredDataset, pivots_dir(root, method, P),
                                    mmap={'graph': True, 'feat': False})
            datapipe = datapipe.zip(pivot_dp)
            datapipe = datapipe.map_args(partial(CollatorPivots, split=split), args=(0,1))
        else:
            datapipe = datapipe.map_args(partial(Collator, split=split))
        # datapipe = datapipe.map(fn=get_samples, output_col=-1)
        index_dp = TensorShuffleWrapper(torch.arange(P))
        datapipe = datapipe.zip(index_dp).collate(collate_fn=list).flat_map(
            partial(even_split_fn, size=batch_size), flatten_col=1)
        datapipe = datapipe.map_args(collate, args=(0, 1))
        num_par = 0 # conf.get('num_workers', max(6, (self.ncpus - self.cpu_i) // 4))
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

class HierarchicalDataLoader(object):
    def __init__(self, dataset: dict, env: dict, split: str, conf: list[dict]):
        self.ctx = mp.get_context('fork')
        self.ncpus = env.get('ncpus', psutil.cpu_count())
        self.cpu_i = 0

        # Level-1 loader, partition-based
        partition_loader = PartitionDataLoader(dataset, env, split, conf[0])
        datapipe = partition_loader.datapipe
        self.cpu_i += partition_loader.cpu_i
        # Level-2 loader, neighbor sampler
        conf_l2 = conf[1]
        fanout = list(map(int, conf[1]['fanout'].split(',')))
        batch_size = conf_l2['batch_size']
        datapipe = datapipe.collate(collate_fn=list).flat_map(
            partial(split_fn, size=batch_size, drop_thres=1), flatten_col=1)
            # partial(even_split_fn, size=batch_size), flatten_col=1)
        num_par = conf_l2.get('num_workers',  self.ncpus - self.cpu_i)
        sample_fn = PygNeighborSampler(fanout)
        datapipe = datapipe.par_map(fn=sample_fn, num_par=num_par, mp_ctx=self.ctx,
                                    affinity=range(self.cpu_i, self.cpu_i+num_par))
        self.cpu_i += num_par

        self.batch_size = (partition_loader.batch_size, batch_size)
        self.datapipe = datapipe

    def __iter__(self):
        return iter(self.datapipe)

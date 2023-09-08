import random, copy, functools
import torch
from datapipe.custom_pipes import IterDataPipe, TensorShuffleWrapper, IterableWrapper
from datapipe.custom_pipes import even_split_fn, shuffle_tensor

def global_batching_dp(examples: torch.Tensor, batch_size, shuffle=False) -> IterDataPipe:
    dp = TensorShuffleWrapper(examples) if shuffle else IterableWrapper([examples])
    return dp.flatmap(fn=functools.partial(even_split_fn, size=batch_size))

from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index
from torch_geometric.sampler.utils import to_csc
from torch_sparse import SparseTensor

def list_shuffle(blocks: list):
    blocks_copy = copy.copy(blocks)
    random.shuffle(blocks_copy)
    return blocks_copy

def hier_batching_dp(
    data: Data,
    blocked: list,
    num_blocks,
    batch_size,
    merge_fn=torch.cat,
) -> IterDataPipe:
    colptr, row, _ = to_csc(data)
    data.adj_t = SparseTensor(rowptr=colptr, col=row)
    data_dp = IterableWrapper([data]).repeats(1000_000)
    block_dp = IterableWrapper([blocked]).map(list_shuffle).unbatch()
    superblock_dp = block_dp.batch(batch_size=num_blocks).map(fn=merge_fn)
    return data_dp.zip(superblock_dp.tensor_shuffle().flatmap(
        fn=functools.partial(even_split_fn, size=batch_size)
    ))

def hier_batching_data_dp(
    data: Data,
    blocked: list,
    num_blocks,
    batch_size,
    merge_fn=torch.cat,
) -> IterDataPipe:
    def subgraph(nodes: torch.Tensor):
        sg = data.subgraph(nodes)
        colptr, row, _ = to_csc(sg)
        sg.adj_t = SparseTensor(rowptr=colptr, col=row)
        return sg
    def train_idx(sg: Data):
        nids = mask_to_index(sg.train_mask)
        return [sg, shuffle_tensor(nids)]
    # block_dp = IterableWrapper(list_shuffle(blocked))
    block_dp = IterableWrapper([blocked]).map(list_shuffle).unbatch()
    superblock_dp = block_dp.batch(batch_size=num_blocks).map(fn=merge_fn)
    return superblock_dp.map(fn=subgraph).map(fn=train_idx).flatmap(
        fn=functools.partial(even_split_fn, size=batch_size), flatten_col=1,
    )

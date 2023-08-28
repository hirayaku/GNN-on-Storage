import torch
from torchdata.datapipes.iter import IterDataPipe
from datapipe.custom_pipes import FullShuffleWrapper, IterableWrapper

def check(x):
    assert len(x) == 1024
    return x

###
# batching for independent examples
###

def global_batching_dp(examples: torch.Tensor, batch_size, shuffle=False) -> IterDataPipe:
    dp = FullShuffleWrapper(examples) if shuffle else IterableWrapper(examples)
    return dp.batch(batch_size=batch_size, wrapper_class=torch.tensor)

def hier_batching_dp(
    blocked,
    num_blocks,
    batch_size,
    shuffle=False,
    merge_fn=torch.cat,
    drop_thres:float=0,
) -> IterDataPipe:
    block_dp = FullShuffleWrapper(blocked) if shuffle else IterableWrapper(blocked)
    superblock_dp = block_dp.batch(batch_size=num_blocks).collate(collate_fn=merge_fn)
    return superblock_dp.in_batch_shuffle2().nested_batch(
        batch_size=batch_size, drop_thres=drop_thres
    )

###
# batching for nodes in a graph
###


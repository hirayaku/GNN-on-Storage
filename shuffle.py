import random
import torch
from datapipe.custom_pipes import IterDataPipe, TensorShuffleWrapper, IterableWrapper

def check(x):
    assert len(x) == 1024
    return x

###
# batching for independent examples
###

def global_batching_dp(examples: torch.Tensor, batch_size, shuffle=False) -> IterDataPipe:
    dp = TensorShuffleWrapper(examples) if shuffle else IterableWrapper([examples])
    return dp.tensor_batch(batch_size=batch_size, drop_last=True)

def hier_batching_dp(
    blocked: list,
    num_blocks,
    batch_size,
    merge_fn=torch.cat,
) -> IterDataPipe:
    shuffled = random.shuffle(blocked)
    block_dp = IterableWrapper(shuffled)
    superblock_dp = block_dp.batch(batch_size=num_blocks).map(fn=merge_fn)
    return superblock_dp.tensor_shuffle().tensor_batch(batch_size=batch_size, drop_last=True)

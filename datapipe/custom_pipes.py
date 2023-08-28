from typing import final, Iterator, Optional, TypeVar, Callable
from functools import partial
import copy
import torch
from torch.utils.data import functional_datapipe, IterDataPipe, DataChunk
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.map import SequenceWrapper

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = [
    'partial',
    'identity_fn',
    'split_fn',
    'even_split_fn',
    'IterableWrapper',
    'LiteIterableWrapper',
    'SequenceWrapper',
    'FullShuffleWrapper',
    'TensorShuffleWrapper',
    'ObjectAsPipe',
]

def identity_fn(*args):
    return args

def shuffle_tensor(tensor: torch.Tensor, generator: Optional[torch.Generator]=None):
    return tensor[torch.randperm(len(tensor), generator=generator)]

def split_fn(input, size: int, drop_thres:float=0):
    min_keep = size * drop_thres
    start = 0
    steps = len(input) // size
    for _ in range(steps):
        end = start + size
        yield input[start:end]
        start = end
    if start < len(input) and len(input) - start >= min_keep:
        yield input[start:]

def even_split_fn(input, size: int):
    remainder = len(input) - len(input) // size * size
    steps = len(input) // size
    surplus = (remainder + steps - 1) // steps 
    start = 0
    for _ in range(steps):
        addition = 0
        if remainder > 0:
            addition = surplus if remainder > surplus else remainder
            remainder -= addition
        end = start + size + addition
        yield input[start:end]
        start = end
    assert start == len(input), f"yielded {start} but input has {len(input)}"

# def nested_batch(batch, inner_batch_size, drop_thres=0.0):
#     if isinstance(batch, tuple):
#         tag = batch[0]
#         data = batch[1]
#     else:
#         tag = None
#         data = batch
#     if tag is None:
#         yield from split_fn(data, inner_batch_size, drop_thres)
#     else:
#         yield from ((tag, s) for s in split_fn(data, inner_batch_size, drop_thres))
# def nested_batch_with_size(size, drop_thres=0.0):
#     return partial(nested_batch, inner_batch_size=size, drop_thres=drop_thres)

class LiteIterableWrapper(IterableWrapper):
    def __init__(self, iterable):
        super().__init__(iterable, deepcopy=False)

class FullShuffleWrapper(IterDataPipe):
    def __init__(self, sequence):
        self._len = len(sequence)
        self.shuffled_dp = SequenceWrapper(sequence).shuffle()

    def __iter__(self):
        return iter(self.shuffled_dp)

    def __len__(self):
        return self._len

class TensorShuffleWrapper(IterDataPipe):
    def __init__(self, tensor: torch.Tensor):
        self._rng: torch.Generator = torch.Generator()
        self._seed: Optional[int] = None
        self._tensor = tensor

    def __len__(self):
        return self._tensor.size(0)

    def __iter__(self):
        # shuffle tensor every epoch
        yield shuffle_tensor(self._tensor, self._rng)

    def reset(self) -> None:
        if self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_(generator=self._rng))
        self._rng.manual_seed(self._seed)
        self._seed = None

    def __len__(self) -> int:
        return len(self._tensor)

    def __getstate__(self):
        # need to attach the tensor to the object: https://github.com/dmlc/dgl/pull/1858
        self._rng_state = self._rng.get_state()
        state = (
            self._tensor,
            self._seed,
            self._rng_state,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        self._tensor, self._seed, self._rng_state = state
        self._rng = torch.Generator()
        self._rng.set_state(self._rng_state)

@functional_datapipe("tensor_batch")
class TensorBatcher(IterDataPipe):
    def __init__(self, tensor_dp, batch_size, drop_last=False, copy=False):
        self.tensor_dp = tensor_dp
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.copy = copy

    def __iter__(self):
        for tensor in self.tensor_dp:
            start = 0
            size = len(tensor)
            batch_size = self.batch_size
            while start + batch_size < size:
                item = tensor[start : start+batch_size]
                if self.copy:
                    item = item.detach().clone()
                yield item
                start += batch_size
            if start < size and not self.drop_last:
                item = tensor[start:]
                if self.copy:
                    item = item.detach().clone()
                yield item

    def __len__(self):
        _len = len(self.tensor_dp) // self.batch_size
        if _len * self.batch_size != len(self.tensor_dp) and not self.drop_last:
            _len += 1
        return _len

@functional_datapipe("tensor_shuffle")
class InBatchTensorShufflerIterDataPipe(IterDataPipe[DataChunk[T_co]]):
    '''
    InBatchShuffler except that
    - each item in the datapipe is a torch tensor
    - torch.Generator used as the RNG
    '''
    def __init__(self, datapipe: IterDataPipe[DataChunk[T_co]]):
        self.datapipe = datapipe
        self._enabled = True
        self._seed: Optional[int] = None
        self._rng: torch.Generator = torch.Generator()

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[DataChunk[T_co]]:
        if not self._enabled:
            for batch in self.datapipe:
                yield batch
        else:
            for batch in self.datapipe:
                idx = torch.randperm(len(batch), generator=self._rng)
                yield batch[idx]

    @final
    def reset(self) -> None:
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self._rng.manual_seed(self._seed)
            self._seed = None

    def __len__(self) -> int:
        return len(self.datapipe)

    def __getstate__(self):
        # need to attach the tensor to the object: https://github.com/dmlc/dgl/pull/1858
        self._rng_state = self._rng.get_state()
        state = (
            self.datapipe,
            self._enabled,
            self._seed,
            self._rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self._enabled,
            self._seed,
            self._rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = torch.Generator()
        self._rng.set_state(self._rng_state)

class ObjectAsPipe(IterDataPipe):
    '''
    lazily initialize an object with cls(*args, **kwrgs),
    and then stream the object out once in a datapipe
    '''
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs
        self._initialized = False
    
    def __iter__(self):
        if not self._initialized:
            self._initialized = True
            self.obj = self.cls(*self.args, **self.kwargs)
        yield self.obj
    
    def __len__(self):
        return 1

# the following pipes are borrowed from torchdata
@functional_datapipe("repeats")
class RepeaterIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, source_datapipe: IterDataPipe[T_co], times: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.times: int = times
        if times <= 1:
            raise ValueError(f"The number of repetition must be > 1, got {times}")

    def __iter__(self) -> Iterator[T_co]:
        for element in self.source_datapipe:
            for _ in range(self.times):
                yield element

    def __len__(self) -> int:
        return self.times * len(self.source_datapipe)

@functional_datapipe("map_args")
class MapperArgsIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    fn: Optional[Callable]

    def __init__(self, datapipe: IterDataPipe, fn: Callable, args=None) -> None:
        self.datapipe = datapipe
        self.fn = fn  # type: ignore[assignment]
        if isinstance(args, list):
            args = tuple(args)
        self.args = args 

    def __iter__(self) -> Iterator[T_co]:
        for d in self.datapipe:
            if self.args is None:
                yield self.fn(d)
            elif isinstance(self.args, tuple):
                args = tuple(d[i] for i in self.args)
                yield self.fn(*args)
            else:
                yield self.fn(d[self.args])

    def __len__(self) -> int:
        return len(self.datapipe)

@functional_datapipe("flat_map")
class FlatMapperIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    fn: Optional[Callable]

    def __init__(self, datapipe: IterDataPipe, fn: Optional[Callable] = None,
                 fn_size: Optional[int] = None, flatten_col=None) -> None:
        self.datapipe = datapipe
        if fn is None:
            fn = identity_fn
        self.fn = fn  # type: ignore[assignment]
        self.fn_size = fn_size
        self.flatten_col = flatten_col

    def __iter__(self) -> Iterator[T_co]:
        if self.flatten_col is None:
            for data in self.datapipe:
                yield from self.fn(data)
        else:
            for data in self.datapipe:
                for item in self.fn(data[self.flatten_col]):
                    data_copy = copy.copy(data)
                    data_copy[self.flatten_col] = item
                    yield data_copy


    def __len__(self) -> int:
        if self.fn_size is None:
            raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")
        else:
            return len(self.datapipe) * self.fn_size

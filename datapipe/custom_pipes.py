from typing import final, Iterator, Optional, TypeVar, Callable, Tuple, Sized, List
from functools import partial
import copy, warnings
import torch
from torch.utils.data.datapipes.iter import UnBatcher
from datapipe import IterDataPipe, make_functional

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

__all__ = [
    'partial',
    'identity_fn',
    'split_fn',
    'even_split_fn',
    'IterableWrapper',
    'LiteIterableWrapper',
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
    steps = max(len(input) // size, 1)
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

# a lot of datapipes below are adapted from torch datapipe implementations
class IterableWrapper(IterDataPipe):
    r"""
    Wraps an iterable object to create an IterDataPipe.

    Args:
        iterable: Iterable object to be wrapped into an IterDataPipe
        deepcopy: Option to deepcopy input iterable object for each
            iterator. The copy is made when the first element is read in ``iter()``.

    .. note::
        If ``deepcopy`` is explicitly set to ``False``, users should ensure
        that the data pipeline doesn't contain any in-place operations over
        the iterable instance to prevent data inconsistency across iterations.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> list(dp)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    def __init__(self, iterable, deepcopy=True):
        self.iterable = iterable
        self.deepcopy = deepcopy

    def __iter__(self):
        source_data = self.iterable
        if self.deepcopy:
            try:
                source_data = copy.deepcopy(self.iterable)
            # For the case that data cannot be deep-copied,
            # all in-place operations will affect iterable variable.
            # When this DataPipe is iterated second time, it will
            # yield modified items.
            except TypeError:
                warnings.warn(
                    "The input iterable can not be deepcopied, "
                    "please be aware of in-place modification would affect source data."
                )
        yield from source_data

    def __len__(self):
        return len(self.iterable)

class LiteIterableWrapper(IterableWrapper):
    def __init__(self, iterable):
        super().__init__(iterable, deepcopy=False)

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

@make_functional("tensor_shuffle")
class InBatchTensorShufflerIterDataPipe(IterDataPipe):
    '''
    InBatchShuffler except that
    - each item in the datapipe is a torch tensor
    - torch.Generator used as the RNG
    '''
    def __init__(self, datapipe: IterDataPipe):
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

    def __iter__(self) -> Iterator:
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

@make_functional('zip')
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators)

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes)
        else:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

# the following pipes are borrowed from torchdata
@make_functional("repeats")
class RepeaterIterDataPipe(IterDataPipe[T_co]):
    def __init__(self, source_datapipe: IterDataPipe[T_co], times: int) -> None:
        self.source_datapipe: IterDataPipe[T_co] = source_datapipe
        self.times: int = times
        if times < 1:
            raise ValueError(f"The number of repetition must be >= 1, got {times}")

    def __iter__(self) -> Iterator[T_co]:
        for element in self.source_datapipe:
            for _ in range(self.times):
                yield copy.copy(element)
            del element

    def __len__(self) -> int:
        return self.times * len(self.source_datapipe)

@make_functional('batch')
class BatcherIterDataPipe(IterDataPipe):
    datapipe: IterDataPipe
    batch_size: int
    drop_last: bool

    def __init__(self,
                 datapipe: IterDataPipe,
                 batch_size: int,
                 drop_last: bool = False,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = tuple

    def __iter__(self):
        batch: List = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield self.wrapper_class(batch)
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield self.wrapper_class(batch)
                batch = []

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

@make_functional('unbatch')
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undoes batching of data (functional name: ``unbatch``). In other words, it flattens the data up to the specified level
    within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self,
                 datapipe: IterDataPipe,
                 unbatch_level: int = 1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            for i in self._dive(element, unbatch_level=self.unbatch_level):
                yield i

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, (list, tuple)):
                for item in element:
                    for i in self._dive(item, unbatch_level=-1):
                        yield i
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, (list, tuple)):
                for item in element:
                    for i in self._dive(item, unbatch_level=unbatch_level - 1):
                        yield i
            else:
                raise IndexError(f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe")


@make_functional("map")
class MapperIterDataPipe(IterDataPipe[T_co]):
    datapipe: IterDataPipe
    fn: Optional[Callable]

    def __init__(self, datapipe: IterDataPipe, fn: Callable, args=None, inplace=False) -> None:
        self.datapipe = datapipe
        self.fn = fn  # type: ignore[assignment]
        if isinstance(args, list):
            args = tuple(args)
        self.args = args 
        self.inplace = inplace

    def __iter__(self) -> Iterator[T_co]:
        if self.args is None:
            for d in self.datapipe:
                yield self.fn(d)
                del d
        elif isinstance(self.args, tuple) or isinstance(self.args, list):
            for d in self.datapipe:
                yield self.fn(*(d[i] for i in self.args))
                del d
        else:
            if self.inplace:
                for d in self.datapipe:
                    d[self.args] = self.fn(d[self.args])
                    yield d
                    del d
            else:
                for d in self.datapipe:
                    yield self.fn(d[self.args])
                    del d

    def __len__(self) -> int:
        return len(self.datapipe)

@make_functional("flatmap")
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
                del data
        else:
            for data in self.datapipe:
                for item in self.fn(data[self.flatten_col]):
                    data_copy = copy.copy(data)
                    data_copy[self.flatten_col] = item
                    yield data_copy
                    del data_copy
                del data

    def __len__(self) -> int:
        if self.fn_size is None:
            raise TypeError(f"{type(self).__name__}'s length relies on the output of its function.")
        else:
            return len(self.datapipe) * self.fn_size

@make_functional("tensor_batch")
class TensorBatcher(IterDataPipe):
    '''
    tensor_batch is just a specialized flatmap
    '''
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
            del tensor

    def __len__(self):
        _len = len(self.tensor_dp) // self.batch_size
        if _len * self.batch_size != len(self.tensor_dp) and not self.drop_last:
            _len += 1
        return _len

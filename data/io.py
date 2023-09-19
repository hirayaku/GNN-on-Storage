import os, json, logging, ctypes, warnings, uuid
from copy import deepcopy
from enum import Enum
from typing import List, Union, Optional
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
from utils import SCRATCH_DIR

logger = logging.getLogger()
_libc = None

class MADV_OPTIONS(Enum):
    MADV_NORMAL       = 0
    MADV_RANDOM       = 1
    MADV_SEQUENTIAL   = 2
    MADV_WILLNEED     = 3
    MADV_DONTNEED     = 4
    MADV_FREE         = 8
    MADV_REMOVE       = 9
    MADV_DONTFORK     = 10
    MADV_DOFORK       = 11
    MADV_MERGEABLE    = 12
    MADV_UNMERGEABLE = 13
    MADV_HUGEPAGE     = 14
    MADV_NOHUGEPAGE  = 15
    MADV_DONTDUMP     = 16
    MADV_DODUMP       = 17
    MADV_WIPEONFORK  = 18
    MADV_KEEPONFORK  = 19
    MADV_COLD        = 20
    MADV_PAGEOUT     = 21
    MADV_POPULATE_READ = 22
    MADV_POPULATE_WRITE = 23
    MADV_DONTNEED_LOCKED = 24
    MADV_HWPOISON     = 100

def madvise(ptr: int, nbytes: int, option: MADV_OPTIONS):
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL('libc.so.6', use_errno=True)
    madvise = _libc.madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    if madvise(ptr, nbytes, option.value) != 0:
        errno = ctypes.get_errno()
        warnings.warn(f"madvise failed with error {errno}: {os.strerror(errno)}")

# an enum of common torch dtypes
class Dtype(Enum):
    bool    = torch.bool
    uint8   = torch.uint8
    int8    = torch.int8
    int16   = torch.int16
    int32   = torch.int32
    int64   = torch.int64
    float16 = torch.float16
    float32 = torch.float32
    float64 = torch.float64
    complex64   = torch.complex64
    complex128  = torch.complex128

    @staticmethod
    def from_torch(dtype: torch.dtype):
        return Dtype(dtype)
    
    @staticmethod
    def from_str(dtype: str):
        return Dtype[dtype]
    
    @staticmethod
    def from_numpy(dtype):
        return Dtype.from_str(str(dtype))

    def to_torch(self) -> torch.dtype:
        return self.value
    
    def to_str(self) -> str:
        return self.name
    
    def to_numpy(self):
        return self.to_str()

def is_tensor(obj) -> bool:
    return isinstance(obj, np.ndarray) \
        or isinstance(obj, torch.Tensor)


class TensorType(Enum):
    '''
    Specifying the physical location of the tensor
    '''
    PlainTensor = 0     # vanilla, in-memory tensor
    MmapTensor = 1      # tensor with a mmap storage
    ShmemTensor = 2     # tensor with a storage in shmem
    PooledTensor = 3    # tensor with a storage in pooled shmem
    DiskTensor = 4      # tensor that primarily lies on disk
    RemoteTensor = 5    # tensor in remote hosts

@dataclass
class TensorMeta:
    shape: List[int]
    dtype: Dtype
    path: Optional[str] = None
    offset: int = 0
    # ro is deprecated due to the undesired interplay between MAP_PRIVATE and MADVISE_DONTNEED
    ro: bool = False
    temporary: bool = False
    random: bool = False
    def __post_init__(self):
        if isinstance(self.dtype, Dtype):
            pass
        elif isinstance(self.dtype, np.dtype):
            self.dtype = Dtype.from_numpy(self.dtype)
        elif isinstance(self.dtype, torch.dtype):
            self.dtype = Dtype.from_torch(self.dtype)
        else:
            self.dtype = Dtype.from_str(self.dtype)
    def _asdict(self):
        return asdict(self)
    def numel(self):
        return torch.prod(torch.LongTensor(self.shape)).item()
    def nbytes(self):
        element_size = torch.tensor([], dtype=self.dtype.to_torch()).element_size()
        return self.numel() * element_size
    def clone(self) -> 'TensorMeta':
        return deepcopy(self)
    def read_only_(self, enable: bool = True) -> 'TensorMeta':
        warnings.warn("'ro' flag is deprecated and not effective any more")
        # self.ro = enable
        return self
    def temp_(self, enable: bool = True) -> 'TensorMeta':
        self.temporary = enable
        return self
    def random_(self, enable: bool = True) -> 'TensorMeta':
        self.random = enable
        return self
    @staticmethod
    def like(tensor: torch.Tensor, dtype=None, path=None):
        return TensorMeta(
            shape=list(tensor.shape),
            dtype=tensor.dtype if dtype is None else dtype,
            path=path,
        )

# ser/des various dtypes <-> string
# https://stackoverflow.com/a/57915246
class DtypeEncoder(json.JSONEncoder):
    def __init__(self, *args, root='.', **kwargs):
        self.rootdir = root
        super(DtypeEncoder, self).__init__(*args, **kwargs)
    def default(self, obj):
        if isinstance(obj, TensorMeta):
            dct = obj._asdict()
            dct['dtype'] = dct['dtype'].to_str()
            dct['path'] = os.path.relpath(dct['path'], start=self.rootdir)
            dct['__tensor_store__'] = True
            return dct
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(DtypeEncoder, self).default(obj)

class DtypeDecoder(json.JSONDecoder):
    def __init__(self, *args, root='.', **kwargs):
        self.rootdir = root
        super(DtypeDecoder, self).__init__(*args, object_hook=self.object_hook, **kwargs)
    def object_hook(self, dct):
        if dct.get('__tensor_store__', False):
            dct['dtype'] = Dtype.from_str(dct['dtype'])
            dct['path'] = os.path.join(self.rootdir, dct['path'])
            dct.pop('__tensor_store__')
            return TensorMeta(**dct)
        return dct


# NOTE: torch.UntypedStorage is formally introduced in pytorch 1.13
# torch.{Type}Storage will be gradually deprecated in future releases
# Most of methods of torch.UntypedStorage are implemented in C++: csrc/Storage*.cpp
# Method definitions in torch/storage.py are used more as type hints
# A key method provided by UntypedStorage is __new__(size: int64_t, allocator: int64_t)
# (csrc/Storage.cpp: THPStorage_pynew). It allows us to provide a customized allocator
# (c10::Allocator *) by torch.UntypedStorage(size=..., allocator=...)
try:
    TorchStorage = torch.UntypedStorage
except AttributeError:
    TorchStorage = torch._UntypedStorage
try:
    t = torch.tensor([])
    t.untyped_storage()
except AttributeError:
    def untyped_storage(self):
        return self.storage().untyped()
    torch.Tensor.untyped_storage = untyped_storage


def MmapTensor(tinfo: TensorMeta, **kwargs) -> torch.Tensor:
    '''
    Create a torch.Tensor backed by an mmap storage specified by TensorMeta
    '''
    for f in fields(tinfo):
        if f in kwargs:
            setattr(tinfo, f, kwargs[f])

    data = torch.tensor([], dtype=tinfo.dtype.to_torch())
    if tinfo.ro and tinfo.path is None:
        raise ValueError("Mmap in read-only mode requires specifying the path")
    if not tinfo.ro:
        if tinfo.path is None:
            fname = f'tensor_{uuid.uuid4().hex}'
            tinfo.path = os.path.join(SCRATCH_DIR, fname)
        elif os.path.isdir(tinfo.path):
            fname = f'tensor_{uuid.uuid4().hex}'
            tinfo.path = os.path.join(tinfo.path, fname)
    allocator = torch.ops.xTensor.mmap_allocator(
        tinfo.path, size=tinfo.nbytes(), read_only=tinfo.ro,
        temp=tinfo.temporary, shm=False)
    storage = TorchStorage(tinfo.nbytes(), allocator=allocator.get())
    data.set_(storage)
    data = data.reshape(tinfo.shape)
    if tinfo.random:
        madvise(data.data_ptr(), data.numel() * data.element_size(), MADV_OPTIONS.MADV_RANDOM)
    data._meta = tinfo.clone()
    return data

def ShmTensor(tinfo: TensorMeta, **kwargs) -> torch.Tensor:
    '''
    Create a torch.Tensor backed by posix shared memory specified by TensorMeta
    '''
    for f in fields(tinfo):
        if f in kwargs:
            setattr(tinfo, f, kwargs[f])
    shared_tensor = torch.tensor([1], dtype=tinfo.dtype.to_torch())
    new_storage = shared_tensor.untyped_storage()._new_shared(tinfo.nbytes())
    new_tensor = shared_tensor.new(new_storage).view(tinfo.shape)
    return new_tensor

'''
class MmapTensor(torch.Tensor):
    @classmethod
    def _new_allocator(cls, tinfo: TensorMeta):
        return torch.ops.xTensor.mmap_allocator(
            tinfo.path, size=tinfo.nbytes(), read_only=tinfo.ro,
            temp=tinfo.temporary, shm=False,
        )

    @classmethod
    def _new_storage(cls, size, allocator):
        return TorchStorage(size, allocator=allocator.get())
    
    def __new__(cls, tinfo: TensorMeta, allocator=None):
        data = torch.tensor([], dtype=tinfo.dtype.to_torch())
        if tinfo.ro and tinfo.path is None:
            raise ValueError("Mmap in read-only mode requires specifying the path")
        if not tinfo.ro:
            if tinfo.path is None:
                fname = f'tensor_{uuid.uuid4().hex}'
                tinfo.path = os.path.join(SCRATCH_DIR, fname)
            elif os.path.isdir(tinfo.path):
                fname = f'tensor_{uuid.uuid4().hex}'
                tinfo.path = os.path.join(tinfo.path, fname)
        if allocator is None:
            allocator = cls._new_allocator(tinfo)
        storage = cls._new_storage(tinfo.nbytes(), allocator)
        data.set_(storage)
        data = data.reshape(tinfo.shape)
        return data.as_subclass(cls)
        # return torch.Tensor._make_subclass(cls, data, data.requires_grad)
    
    def __init__(self, tinfo: TensorMeta, allocator=None):
        self._tinfo = tinfo
        self._ttype = TensorType.MmapTensor
        if tinfo.random:
            madvise_random(self.data_ptr(), self.numel() * self.element_size())
    
    def upcast(self) -> torch.Tensor:
        data = torch.tensor([], dtype=self.dtype)
        data.set_(self.untyped_storage(), self.storage_offset(), self.shape, self.stride())
        return data

class ShmemTensor(torch.Tensor):
    @classmethod
    def _new_allocator(cls, shm_name, size):
        return torch.ops.xTensor.libshm_allocator(
            shm_name, size=size, # don't care about the rest of tinfo
        )

    @classmethod
    def _new_storage(cls, size, allocator):
        return TorchStorage(size, allocator=allocator.get())

    def __new__(cls, tinfo: TensorMeta, allocator=None):
        if tinfo.ro:
            raise ValueError("ShmemTensor doesn't support read-only")
        if not tinfo.temporary:
            raise ValueError("ShmemTensor can't guarantee to be persistent")
        data = torch.tensor([], dtype=tinfo.dtype.to_torch())
        if allocator is None:
            if tinfo.path is None:
                tinfo.path = f'tensor_{uuid.uuid4().hex}'
            allocator = cls._new_allocator(tinfo.path, tinfo.nbytes())
        storage = cls._new_storage(tinfo.nbytes(), allocator)
        data.set_(storage)
        data = data.reshape(tinfo.shape)
        return data.as_subclass(cls)
        # return torch.Tensor._make_subclass(cls, data, data.requires_grad)
    
    def __init__(self, tinfo: TensorMeta, allocator=None):
        self._tinfo = tinfo
        self._ttype = TensorType.ShmemTensor

    def upcast(self) -> torch.Tensor:
        data = torch.tensor([], dtype=self.dtype)
        data.set_(self.untyped_storage(), self.storage_offset(), self.shape, self.stride())
        return data
    
import multiprocessing
import torch.multiprocessing.reductions as torch_reductions

# copied from pytorch multiprocessing reductions.py
def reduce_storage_fd(storage: TorchStorage):
    if storage.size() == 0:
        return (torch_reductions.rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        # cache_key = torch_reductions.fd_id(fd) # omit caching on the sender side
        metadata = (df, size)
        return (torch_reductions.rebuild_storage_fd, (type(storage),) + metadata)

def reduce_storage_fs(storage: TorchStorage):
    if storage.size() == 0:
        return (torch_reductions.rebuild_storage_empty, (type(storage),))
    else:
        metadata = storage._share_filename_cpu_()
        # cache_key = metadata[1]
        rebuild = torch_reductions.rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
        return (rebuild, (type(storage),) + metadata)

def reduce_storage(storage: TorchStorage):
    # override reduce_storage for custom tensors
    if hasattr(storage, '_ttype'):
        if storage._ttype == TensorType.MmapTensor:
            return reduce_storage_fd(storage)
        elif storage._ttype == TensorType.ShmemTensor:
            return reduce_storage_fs(storage)
    return torch_reductions.reduce_storage(storage)

def rebuild_custom_tensor(cls, untyped_storage, metadata):
    dtype, storage_offset, size, stride, requires_grad = metadata
    flat = torch.empty(0, dtype=dtype)
    flat.set_(untyped_storage)
    t = torch.empty(0, dtype=dtype)
    t.set_(flat, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # we have to pass requires_grad into constructor, rather than set it as an
        # attribute later, because it's an important check for Integer Tensors to
        # have requires_grad=False (or else they raise an error)
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t

def reduce_mmap_tensor_fd(tensor: 'MmapTensor'):
    # print("CUSTOM recuder of MmapTensor")
    assert isinstance(tensor, MmapTensor)
    storage = tensor.untyped_storage()
    # tag the storage to invoke our own reducer
    storage._ttype = tensor._ttype
    metadata = (
        tensor.dtype, tensor.storage_offset(), tensor.size(),
        tensor.stride(), tensor.requires_grad
    )
    return (
        rebuild_custom_tensor,
        (type(tensor),
         storage,
         metadata)
    )

def reduce_shmem_tensor_fs(tensor: 'ShmemTensor'):
    # print("CUSTOM reducer on ShmemTensor")
    assert isinstance(tensor, ShmemTensor)
    storage = tensor.untyped_storage()
    # tag the storage to invoke our own reducer
    storage._ttype = tensor._ttype
    metadata = (
        tensor.dtype, tensor.storage_offset(), tensor.size(),
        tensor.stride(), tensor.requires_grad
    )
    return (
        rebuild_custom_tensor,
        (type(tensor),
         storage,
         metadata)
    )

### register custom tensor classes into the ForkingPickler ###
# _ForkingPickler = multiprocessing.reduction.ForkingPickler
# _ForkingPickler.register(TorchStorage, reduce_storage)
# _ForkingPickler.register(MmapTensor, reduce_mmap_tensor_fd)
# _ForkingPickler.register(ShmemTensor, reduce_shmem_tensor_fs)
###
'''

def load_tensor(
    tinfo: TensorMeta,
    ttype: TensorType = TensorType.PlainTensor,
) -> torch.Tensor:
    # don't create files if it doesn't exist
    size = torch.prod(torch.LongTensor(tinfo.shape)).item()
    if ttype is TensorType.PlainTensor:
        array = np.fromfile(tinfo.path, dtype=tinfo.dtype.to_str(), count=size)
        return torch.from_numpy(array).reshape(tinfo.shape)
    elif ttype is TensorType.MmapTensor:
        assert tinfo.path and os.path.exists(tinfo.path), \
            f"Invalid file path {tinfo.path}"
        return MmapTensor(tinfo)
    # elif ttype is TensorType.ShmemTensor:
    #     mmapped = load_tensor(tinfo, TensorType.MmapTensor)
    #     tensor = ShmemTensor(tinfo, backend)
    #     tensor.copy_(mmapped)
    #     return tensor
    else:
        raise NotImplementedError(f"TensorType {ttype.name} not supported yet")

def store_tensor(
    tensor: Union[np.ndarray, torch.Tensor], path: str
) -> TensorMeta:
    if isinstance(tensor, np.ndarray):
        tensor.tofile(path)
        meta = TensorMeta(
            shape=tensor.shape,
            dtype=tensor.dtype,
            path=path, offset=0
        )
    elif isinstance(tensor, torch.Tensor):
        if hasattr(tensor, '_meta') and tensor._meta.temporary is False:
            meta = tensor._meta
        else:
            tensor.numpy().tofile(path)
            meta = TensorMeta(
                shape=tensor.shape,
                dtype=tensor.dtype,
                path=path, offset=0
            )
    else:
        tinfo = tensor.save(path)
        meta = TensorMeta(
            shape=tinfo.shape,
            dtype=tinfo.dtype,
            path=tinfo.path, offset=tinfo.offset
        )
    # prevent auto deleting the file when next time loading it
    meta.temporary = False
    logger.info(f"tensor saved to: {meta.path}")
    return meta

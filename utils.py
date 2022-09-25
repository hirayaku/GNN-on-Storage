import sys, os, ctypes, mmap
import numpy as np
import torch

libc = ctypes.CDLL('libc.so.6', use_errno=True)

def madvise_random(ptr: int, nbytes: int):
    madvise = libc.madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int
    if madvise(ptr, nbytes, 1) != 0:
        errno = ctypes.get_errno()
        print(f"madvise failed with error {errno}: {os.strerror(errno)}")
        sys.exit(errno)

class DiskTensor(np.memmap):
    '''
    An adaptor class for np.memmap to be compatible with DGLGraph node tensors.
    Accesses to the underlying data rely on the mmap mechanism (page faults, prefetches, etc.),
    and are single-threaded because of numpy. It might not be very fast for
    large amounts of random accesses.
    '''
    def __new__(cls, filename, dtype=np.uint8, mode='r+', offset=0,
                shape=None, order='C'):
        obj = super().__new__(cls, filename, dtype=dtype, mode=mode,
            offset=offset, shape=shape, order=order)
        return obj

    def madvise_random(self):
        try:
            self._mmap.madvise(mmap.MADV_RANDOM)
        except AttributeError:
            # in python<3.8 mmap doesn't provide madvise
            # https://github.com/numpy/numpy/issues/13172
            madvise = libc.madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int
            if madvise(self.ctypes.data, self.size * self.dtype.itemsize, 1) != 0:
                errno = ctypes.get_errno()
                print(f"madvise failed with error {errno}: {os.strerror(errno)}")
                sys.exit(errno)

    def to(self, device):
        return torch.tensor(self, device=device)
        #return torch.tensor(self, dtype=torch.float32, device=device)

def memmap(path, random=False, dtype=np.int8, mode='r+', offset=0, shape=None, order='C'):
    '''
    memmap binary files, can be tuned for random accesses
    '''
    data = DiskTensor(path, dtype=dtype, mode=mode, offset=offset,
        shape=tuple(shape), order=order)
    if random:
        data.madvise_random()
    return data

def mask2index(mask: torch.BoolTensor):
    return torch.nonzero(mask, as_tuple=True)[0]

# serialize various dtypes to string
import json
# https://stackoverflow.com/a/57915246
class DtypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj).split('.')[-1]
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(DtypeEncoder, self).default(obj)

# lookup torch dtype from numpy dtype
numpy_to_torch_dtype_dict = {
    np.dtype('bool')       : torch.bool,
    np.dtype('uint8')      : torch.uint8,
    np.dtype('int8')       : torch.int8,
    np.dtype('int16')      : torch.int16,
    np.dtype('int32')      : torch.int32,
    np.dtype('int64')      : torch.int64,
    np.dtype('float16')    : torch.float16,
    np.dtype('float32')    : torch.float32,
    np.dtype('float64')    : torch.float64,
    np.dtype('complex64')  : torch.complex64,
    np.dtype('complex128') : torch.complex128
}

def torch_dtype(dtype):
    return numpy_to_torch_dtype_dict[np.dtype(dtype)]

from contextlib import contextmanager
@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    print('%s: mem=%s MB' % (point, usage[2]/1024.0 ))


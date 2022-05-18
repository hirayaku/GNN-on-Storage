import sys, os, ctypes, mmap
import numpy as np
import torch

libc = ctypes.CDLL('libc.so.6', use_errno=True)

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
        # for compatiblity with DGLGraph
        obj.device = torch.device('cpu')
        return obj

    def __array_finalize__(self, obj):
        self.device = getattr(obj, 'device', None)

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


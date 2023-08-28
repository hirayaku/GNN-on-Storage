import unittest
import time, os
import torch
import utils
import data.io
from data.io import Dtype, TensorMeta, TensorType, MmapTensor, ShmemTensor, load_tensor

class TestCustomTensor(unittest.TestCase):
    def setUp(self):
        super().setUp()
        print()

    def test_mmap_tensor(self):
        tinfo = TensorMeta(
            shape=(4, 4),
            dtype=Dtype.int64,
            # path=None,
            ro=False,
            temporary=False,
            random=True,
        )
        mmap_tensor = MmapTensor(tinfo)
        assert mmap_tensor.is_shared()
        print("generated file:", tinfo.path)
        print(mmap_tensor)
        mmap_tensor[0, :] = torch.arange(mmap_tensor.size(1))
        print(mmap_tensor)

        del mmap_tensor
        tensor = load_tensor(tinfo)
        assert (tensor[0, :] == torch.arange(tensor.size(1))).all()

    def test_temp_mmap_tensor(self):
        tinfo = TensorMeta(
            shape=(4, 4),
            dtype=Dtype.int64,
            # path=None,
            ro=False,
            temporary=True,
            random=True,
        )
        mmap_tensor = MmapTensor(tinfo)
        assert mmap_tensor.is_shared()
        print(mmap_tensor)
        mmap_tensor[0, :] = torch.arange(mmap_tensor.size(1))
        print(mmap_tensor)
        assert not os.path.exists(tinfo.path)
    
    def test_shm_tensor(self):
        tinfo = TensorMeta(
            shape=(4, 4),
            dtype=Dtype.int64,
            # path=None,
            ro=False,
        )
        shm_tensor = ShmemTensor(tinfo)
        assert shm_tensor.is_shared()
        print("generated file:", tinfo.path)
        print(shm_tensor)
        shm_tensor[0, :] = torch.arange(shm_tensor.size(1))
        print(shm_tensor)
        print('/dev/shm:', os.listdir('/dev/shm'))
        assert os.path.exists(os.path.join('/dev/shm', tinfo.path))

    def test_temp_shm_tensor(self):
        tinfo = TensorMeta(
            shape=(4, 4),
            dtype=Dtype.int64,
            path='shm_test',
            ro=False,
            temporary=True
        )
        shm_tensor = ShmemTensor(tinfo)
        assert shm_tensor.is_shared()
        print("generated file:", tinfo.path)
        print(shm_tensor)
        shm_tensor[0, :] = torch.arange(shm_tensor.size(1))
        print(shm_tensor)
        shm_tensor = MmapTensor(tinfo)
        assert not os.path.exists(os.path.join('/dev/shm', tinfo.path))

    def test_large_shm(self):
        # allocate close to 32GB in shm. If allocating exactly 32GB,
        # it report "Bus Error" on my machine
        # I suspect it's because something else is taking up space in /dev/shm
        tinfo = TensorMeta(
            shape=(int(1024*1024*3.9), 1024),
            dtype=Dtype.int64,
            # path=None,
            ro=False,
        )
        shm_tensor = ShmemTensor(tinfo)
        assert shm_tensor.numel() == tinfo.numel(), \
            f"Allocated {shm_tensor.numel()} but requested {tinfo.numel()}"
        assert shm_tensor.is_shared()
        print("generated file:", tinfo.path)
        print(shm_tensor.shape)
        print(shm_tensor)
        for i in range(shm_tensor.size(0)):
            if i % (1024*1024) == 0:
                print("Filled", i, "rows")
            shm_tensor[i, :] = torch.arange(shm_tensor.size(1))
        print(shm_tensor)
    
    def test_ro_mmap_nofile(self):
        tinfo = TensorMeta(
            shape=(4, 4),
            dtype=Dtype.int64,
            # path=None,
            ro=True,
            # temporary=True,
            # random=True,
        )
        with self.assertRaises((RuntimeError, ValueError)):
            MmapTensor(tinfo)
    
    def test_index_select_lc(self):
        tinfo = TensorMeta(
            shape=(8, 8),
            dtype=Dtype.int64,
            # path=None,
            ro=False,
            # temporary=True,
            # random=True,
        )
        src = MmapTensor(tinfo)
        assert src.is_shared()
        src.copy_(torch.randint(0, 1024, src.shape))
        print(src)

        out = MmapTensor(tinfo)
        assert out.is_shared()
        perm = torch.randperm(src.size(0))
        data.io.index_select(src, perm, out=out, buf_size=8)
        print(perm)
        print(out)

if __name__ == "__main__":
    unittest.main()

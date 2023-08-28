# NOTE: this file is deprecated
import unittest
import torch
import torch.multiprocessing as torch_mp
from data.io import Dtype, TensorMeta, MmapTensor
from data.shmem import ShmemTensor

def reader_fname(wid: int, tensor: torch.Tensor):
    print("After sharing:")
    torch.random.manual_seed(int(tensor[0][0]))
    torch.randint(0, 64, tensor.size(), out=tensor)
    print(tensor)
    # print("Fd =", tensor.untyped_storage()._get_shared_fd())
    # print("Fname =", tensor.untyped_storage()._share_filename_cpu_())

def reader_fd(wid: int, tensor: torch.Tensor):
    print("After sharing:")
    torch.random.manual_seed(int(tensor[0][0]))
    torch.randint(0, 64, tensor.size(), out=tensor)
    print(tensor)
    # print("Fd =", tensor.untyped_storage()._get_shared_fd())

class TestShareBuiltinTensor(unittest.TestCase):
    def setUp(self):
        super().setUp()
        print()

    def test_share_with_filesystem(self):
        torch_mp.set_sharing_strategy('file_system')
        tensor = torch.randint(0, 64, (4, 4), dtype=torch.long)
        tensor.share_memory_()
        print("Before sharing:")
        # print("Fname =", tensor.untyped_storage()._share_filename_cpu_())
        print(tensor)

        torch_mp.set_sharing_strategy('file_system')
        print("Sharer #1")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharer #2")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharing completes")
        print(tensor)

    def test_share_with_fd(self):
        torch_mp.set_sharing_strategy('file_descriptor')
        tensor = torch.randint(0, 64, (4, 4), dtype=torch.long)
        tensor.share_memory_()
        print("Before sharing:")
        # print("Fd =", tensor.untyped_storage()._get_shared_fd())
        print(tensor)

        print("Sharer #1")
        sharer = torch_mp.spawn(fn=reader_fd, args=(tensor,))
        print("Sharer #2")
        sharer = torch_mp.spawn(fn=reader_fd, args=(tensor,))
        print("Sharing completes")
        print(tensor)


class TestShareCustomTensor(unittest.TestCase):
    def setUp(self):
        super().setUp()
        print()

    def test_share_mmap_with_filesystem(self):
        torch_mp.set_sharing_strategy('file_descriptor')
        tinfo = TensorMeta(
            shape=[4, 4],
            dtype=Dtype.int64,
            path='tmp_file',
            ro=False,
            temporary=True,
        )
        tensor = MmapTensor(tinfo)
        torch.randint(0, 64, tensor.size(), out=tensor)
        print("Before sharing:")
        print(tensor)
        print("Fd =", tensor.untyped_storage()._get_shared_fd())

        # NOTE: our mmap tensor is created with MapAllocator, which only works with the 
        # file_descriptor sharing strategy. Since we switch to the file_system strategy now,
        # we can't directly share a MapAllocator storage. So, a new storage is created to
        # replace the original storage.
        torch_mp.set_sharing_strategy('file_system')
        print("Sharer #1")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharer #2")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharing completes")
        print(tensor)
        # The new storage doesn't keep fd, causing the exception.
        print("Fd =", tensor.untyped_storage()._get_shared_fd())

    def test_share_shm_with_filesystem(self):
        torch_mp.set_sharing_strategy('file_system')
        tinfo = TensorMeta(
            shape=[4, 4],
            dtype=Dtype.int64,
            path='tmp_file',
            ro=False,       # must set to False for ShmemTensor
        )
        tensor = ShmemTensor(tinfo)
        torch.randint(0, 64, tensor.size(), out=tensor)
        print("Before sharing:")
        print(tensor)
        print("Fname =", tensor.untyped_storage()._share_filename_cpu_())

        print("Sharer #1")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharer #2")
        sharer = torch_mp.spawn(fn=reader_fname, args=(tensor,))
        print("Sharing completes")
        print(tensor)

    def test_share_with_fd(self):
        torch_mp.set_sharing_strategy('file_descriptor')
        tinfo = TensorMeta(
            shape=[4, 4],
            dtype=Dtype.int64,
            path='tmp_file',
            ro=False,
            temporary=True, # file is immediately unlinked after mmap
        )
        tensor = MmapTensor(tinfo)
        torch.randint(0, 64, tensor.size(), out=tensor)
        print("Before sharing:")
        print(tensor)
        print("Fd =", tensor.untyped_storage()._get_shared_fd())

        print("Sharer #1")
        sharer = torch_mp.spawn(fn=reader_fd, args=(tensor,))
        print("Sharer #2")
        sharer = torch_mp.spawn(fn=reader_fd, args=(tensor,))
        print("Sharing completes")
        print(tensor)
        print("Fd =", tensor.untyped_storage()._get_shared_fd())

def dump(i, queue):
    data = queue.get()
    print(data)
    print(f"Worker[{i}], shared: {data.is_shared()}, pinned: {data.is_pinned()}")

class TestQueueTensor(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ctx = torch_mp.get_context('spawn')
        self.queue = self.ctx.SimpleQueue()
        print()

    def test_tensor(self):
        tensor = torch.randint(0, 1024, (4,4))
        print(tensor)
        worker = torch_mp.spawn(fn=dump, args=(self.queue,), join=False)
        self.queue.put(tensor)
        worker.join()
        print(f"Leader, shared:", tensor.is_shared())
    
    def test_mmap_tensor(self):
        # it shouldn't matter to our custom MmapTensor
        torch_mp.set_sharing_strategy('file_system')
        tinfo = TensorMeta(
            shape=[4, 4],
            dtype=Dtype.int64,
            path='tmp_file',
            ro=False,
            temporary=True,
        )
        tensor = MmapTensor(tinfo)
        torch.randint(0, 64, tensor.size(), out=tensor)
        print(tensor)
        print(f"Leader, shared:", tensor.is_shared())
        worker = torch_mp.spawn(fn=dump, args=(self.queue,), join=False)
        self.queue.put(tensor)
        worker.join()
    
    def test_pinned_tensor(self):
        tensor = torch.randint(0, 1024, (4,4), pin_memory=True)
        print(tensor)
        print(f"shared: {tensor.is_shared()}, pinned: {tensor.is_pinned()}")
        worker = torch_mp.spawn(fn=dump, args=(self.queue,), join=False)
        self.queue.put(tensor)
        worker.join()

    def test_allocator(self):
        # allocator = torch.ops.xTensor.default_cpu_allocator()
        allocator = torch.ops.xTensor.libshm_allocator('tmp_file', 1024)
        print(allocator)
        self.queue.put(allocator)
        worker = torch_mp.spawn(fn=dump, args=(self.queue,))

if __name__ == "__main__":
    # ctx = torch_mp.get_context('spawn')
    # queue = ctx.SimpleQueue()
    test = unittest.TestSuite()
    test.addTest(TestShareCustomTensor('test_share_mmap_with_filesystem'))
    unittest.TextTestRunner().run(test)
    # unittest.main()

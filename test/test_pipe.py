import os, time, sys, logging, unittest, tqdm
from functools import partial
import torch
import torchdata # NOTE: for prefetch
from data.graphloader import *
from datapipe.custom_pipes import IterableWrapper, LiteIterableWrapper, identity_fn, split_fn
from datapipe.parallel_pipes import mp, make_dp_worker
from trainer.dataloader import NodeDataLoader, NodeTorchDataLoader, PartitionDataLoader, HierarchicalDataLoader

logger = logging.getLogger()
logger.level = logging.DEBUG

def probe(item):
    print(item)
    return item

def fn(tensor: torch.Tensor):
    time.sleep(0.5)
    tensor *= 2
    return tensor


class TestPipe(unittest.TestCase):
    def setUp(self):
        super().setUp()
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
            datefmt='%0y-%0m-%0d %0H:%0M:%0S')
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler.setFormatter(formatter)
        logger.addHandler(self.stream_handler)
        print()

    def tearDown(self):
        logger.removeHandler(self.stream_handler)
    
    def test_wrapper(self):
        ones = [1] * 100_000
        dp = IterableWrapper(ones).map(identity_fn)
        for _ in dp:
            pass
    
    def test_pmap1(self):
        spawn_ctx = mp.get_context('spawn')
        fork_ctx = mp.get_context('fork')
        num_workers = 4
        tensor = torch.arange(128)
        tensor.share_memory_()
        dp1 = IterableWrapper([tensor]).tensor_batch(4)
        dp2 = dp1.fork(1)
        dp1 = make_dp_worker(dp1, spawn_ctx)
        dp2 = make_dp_worker(dp2, spawn_ctx)
        # we can mix spawn and fork here to save memory
        dp1 = dp1.par_map(fn, num_par=num_workers, mp_ctx=fork_ctx)
        dp2 = dp2.par_map(fn, num_par=num_workers, mp_ctx=fork_ctx)
        dp1.name = "dp1"
        dp2.name = "dp2"
        dp = dp1.zip(dp2)

        for _ in range(2):
            now = time.time()
            for b in dp:
                print(b)
            print("done:", time.time() - now, "s")
    
    def test_pmap2(self):
        dp = LiteIterableWrapper([torch.arange(8*200)]).flat_map(partial(split_fn, size=8))
        dp = dp.par_map(identity_fn, num_par=4, mp_ctx=mp.get_context('fork'))
        for _ in range(3):
            count = 0
            for batch in dp:
                count += 1 
            print("Generated", count, "items")

    def test_pypeln(self):
        import pypeln as pl
        num_workers = 4
        tensor = torch.arange(128)
        tensor.share_memory_()
        dp = IterableWrapper([tensor]).tensor_batch(4)
        dp = pl.process.map(fn, dp, workers=num_workers, maxsize=num_workers)

        for _ in range(2):
            now = time.time()
            for b in dp:
                print(b)
            print("done:", time.time() - now, "s")
 
    # XXX the datapipe doesn't release memory in time
    def test_psample(self):
        dataloader = PartitionDataLoader(
            dataset={'root': '/mnt/md0/hb_datasets/ogbn_papers100M'},
            env={'ncpus': 24},
            split='train',
            conf={
                "sampler": "cluster",
                "partition": "fennel-lb",
                "P": 1024,
                "batch_size": 1024//8,
                "pivots": True,
                "num_workers": 8,
            },
        )
        for e in range(3):
            now = time.time()
            logger.info(f"Epoch {e}")
            for _ in tqdm.tqdm(dataloader):
                pass
            print(f"Epoch {e} done:", time.time() - now, "s")
    
    # XXX 1. when num_intra_par == 2, dataloader gets stuck
    # FIXED 2. par_map: some data are dropped when num_workers > 1 (mp.queue is not in-order)
    # 3. using par_map is much slower than the torch DataLoader
    def test_nsample(self):
        root = '/mnt/md0/hb_datasets/ogbn_products'
        dataloader = NodeDataLoader(
            dataset={'root': root},
            env={'ncpus': 24},
            split='train',
            conf={
                "sampler": "ns",
                "batch_size": 1000,
                "fanout": "15,10,5",
                "num_workers": 1,
            },
        )
        for e in range(3):
            now = time.time()
            edges = 0
            iters = 0
            logger.info(f"Epoch {e}")
            for batch in tqdm.tqdm(dataloader):
                edges += batch.adj_t.nnz()
                iters += 1
                # print(batch)
            print(f"Epoch {e} done: {time.time() - now:.2f}s, {iters} batches, #avg-edges:", edges//iters)

    def test_torch_nsample(self):
        root = '/mnt/md0/hb_datasets/ogbn_products'
        dataloader = NodeTorchDataLoader(
            dataset={'root': root},
            env={'ncpus': 24},
            split='train',
            conf={
                "sampler": "ns",
                "batch_size": 1000,
                "fanout": "15,10,5",
                "num_workers": 0,
            },
        )
        time.sleep(1)
        for e in range(3):
            now = time.time()
            edges = 0
            iters = 0
            logger.info(f"Epoch {e}")
            for batch in tqdm.tqdm(dataloader):
                edges += batch.adj_t.nnz()
                iters += 1
            print(f"Epoch {e} done: {time.time() - now:.2f}s. #avg-edges:", edges//iters)
    
    def test_hbsample(self):
        dataloader = HierarchicalDataLoader(
            dataset={'root': '/mnt/md0/hb_datasets/ogbn_papers100M'},
            env={'ncpus': 24},
            split='train',
            conf=[
                {
                    "sampler": "cluster",
                    "partition": "fennel-lb",
                    "P": 1024,
                    "batch_size": 128,
                    "pivots": True,
                    "num_workers": 0,
                }, {
                    "sampler": "ns",
                    "batch_size": 1000,
                    "fanout": "15,10,5",
                    "num_workers": 12,
                },
            ],
        )
        for e in range(1):
            now = time.time()
            edges, iters = 0, 0
            logger.info(f"Epoch {e}")
            for batch in tqdm.tqdm(dataloader):
                edges += batch.adj_t.nnz()
                iters += 1
            print(f"Epoch {e} done: {time.time() - now:.2f}s. #avg-edges:", edges//iters)

if __name__ == "__main__":
    # test = unittest.TestSuite()
    # test.addTest(TestPipe('test_nsample'))
    # unittest.TextTestRunner().run(test)
    unittest.main()

import time, sys, os, logging, unittest
import torch
logger = logging.getLogger()
logger.level = logging.DEBUG

class TestCudaStream(unittest.TestCase):
    def setUp(self):
        super().setUp()
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
            datefmt='%0y-%0m-%0d %0H:%0M:%0S')
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler.setFormatter(formatter)
        logger.addHandler(self.stream_handler)
        print()

    def test_stream(self):
        s = torch.cuda.Stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            cpu_start = time.time()
            with torch.cuda.stream(s):
                start.record()
                t1 = torch.rand((1024*32, 1024*32), device='cuda')
                t2 = torch.rand((1024*32, 1024*32), device='cuda')
                t1 = t1.to('cuda', non_blocking=True)
                t1 = t1.mm(t2).mm(t2)
                end.record()
            cpu_end = time.time()
            logger.info(f"took {cpu_end-cpu_start:.3f}s on CPU")
            s.synchronize()
            logger.info(f"took {start.elapsed_time(end)/1000:.3f} on GPU")

if __name__ == "__main__":
    unittest.main()

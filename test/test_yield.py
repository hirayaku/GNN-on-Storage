import os, time, sys, logging, unittest, objgraph
import torch
from torch.utils.data.datapipes.iter import Mapper
from datapipe.custom_pipes import IterableWrapper, identity_fn
import utils
logger = logging.getLogger()
logger.level = logging.DEBUG

def gen_tensor_inner(n):
    for _ in range(n):
        yield torch.randint(1024, (1024*1024*1024//8,))

def gen_tensor(n):
    g = gen_tensor_inner(n)
    yield from g

class TestYield(unittest.TestCase):
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
    
    def test_yield(self):
        gen = gen_tensor(4)
        utils.report_mem(point='init')
        for t in gen:
            utils.report_mem(point='before')
            del t
            utils.report_mem(point='after')
    
    def test_pipe(self):
        gen = gen_tensor(4)
        dp = IterableWrapper(gen)
        utils.report_mem(point='init')
        for d in dp:
            utils.report_mem(point='before')
            objgraph.show_backrefs(d, max_depth=5, filename='backrefs.png')
            del d
            utils.report_mem(point='after')

if __name__ == "__main__":
    unittest.main()

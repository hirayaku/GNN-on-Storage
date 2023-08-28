import os, unittest
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import mask_to_index, degree
from data.ops import scatter, scatter_append
from data.graphloader import NodePropPredDataset, ChunkedNodePropPredDataset
from data.collater import Collator, CollatorPivots

import sys, logging
logger = logging.getLogger()
logger.level = logging.DEBUG

class TestCollate(unittest.TestCase):
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

    def test_products(self):
        dataset = 'ogbn-products'
        normalize = dataset.replace('-', '_')
        root = '/mnt/md0/hb_datasets'
        logger.info("Collate ogbn-products")
        self.baseline = NodePropPredDataset(
            dataset, f'{root}/{normalize}', mmap=(True, True),
        )
        # print(self.baseline.num_nodes, self.baseline[0].edge_index[0].size(0))
        self.partitions = ChunkedNodePropPredDataset(
            dataset, f'{root}/{normalize}/fennel-lb-P64', mmap=(True, True),
        )
        self.P = self.partitions.N_P
        self.check_edge = True
        self.collator = Collator(self.partitions, merge_nid=True)
        self.run_collate()

    def test_arxiv(self):
        dataset = 'ogbn-arxiv'
        normalize = dataset.replace('-', '_')
        root = '/mnt/md0/hb_datasets'
        logger.info("Collate ogbn-arxiv")
        self.baseline = NodePropPredDataset(
            dataset, f'{root}/{normalize}', mmap=(True, True),
        )
        self.partitions = ChunkedNodePropPredDataset(
            dataset, f'{root}/{normalize}/fennel-lb-P64', mmap=(True, True),
        )
        self.P = self.partitions.N_P
        self.check_edge = True
        self.collator = Collator(self.partitions, merge_nids=True)
        for _ in range(8):
            self.run_collate()

    def test_papers(self):
        dataset = 'ogbn-papers100M'
        normalize = dataset.replace('-', '_')
        root = '/mnt/md0/hb_datasets'
        logger.info("Collate ogbn-papers100M")
        self.baseline = NodePropPredDataset(
            dataset, f'{root}/{normalize}', mmap=(True, True),
        )
        self.partitions = ChunkedNodePropPredDataset(
            dataset, f'{root}/{normalize}/fennel-lb-P1024', mmap=(True, True),
        )
        self.P = self.partitions.N_P
        self.merge_nid = True
        self.collator = Collator(self.partitions, self.merge_nid)
        for _ in range(1):
            self.run_collate()

    def run_collate(self):
        batch = torch.randperm(self.P)[:self.P//8]
        sub_data, _ = self.collator.collate(batch)
        sub_index = sub_data.edge_index
        sub_index = SparseTensor(row=sub_index[1], col=sub_index[0], is_sorted=True)

        if not self.collator.merge_nid:
            relabel_nodes = self.collator.batch_nodes(batch)
            nodes = self.partitions.node_map[relabel_nodes]
            baseline_index = self.baseline[0].edge_index
            baseline_sp = SparseTensor(row=baseline_index[1], col=baseline_index[0], is_sorted=True)
            baseline_index = baseline_sp[nodes, nodes]
            assert sub_index.numel() == baseline_index.numel(), "different #edges"
            assert (self.partitions.node_map[sub_index.coo()[0]] == \
                nodes[baseline_index.coo()[0]]).all(), "different edge sources"

class TestCollatePivots(TestCollate):

    def test_arxiv(self):
        dataset = 'ogbn-arxiv'
        root = '/mnt/md0/hb_datasets'
        self.load_dataset(dataset, root, 64)
        BM = 8
        for _ in range(BM):
            self.run_collate(BM)

    def test_products(self):
        dataset = 'ogbn-products'
        root = '/mnt/md0/hb_datasets'
        self.load_dataset(dataset, root, 64)
        BM = 1
        for _ in range(BM):
            self.run_collate(BM, check=True)

    def test_papers(self):
        dataset = 'ogbn-papers100M'
        root = '/mnt/md0/hb_datasets'
        self.load_dataset(dataset, root, 1024)
        BM = 8
        for _ in range(BM):
            self.run_collate(BM, check=True)
    
    def load_dataset(self, dataset, root, pn):
        normalize = dataset.replace('-', '_')
        logger.info(f"Collate {dataset}")
        self.baseline = NodePropPredDataset(
            f'{root}/{normalize}', mmap=(True, True),
        )
        self.partitions = ChunkedNodePropPredDataset(
            f'{root}/{normalize}/fennel-lb-P{pn}', mmap=(True, True),
        )
        self.pivots = ChunkedNodePropPredDataset(
            f'{root}/{normalize}/fennel-lb-P{pn}-pivots', mmap=(True, False),
        )
        self.P = self.partitions.N_P
        self.collator = CollatorPivots(self.partitions, self.pivots)

    def run_collate(self, BM:int, check:bool=False):
        batch = torch.randperm(self.P)[:self.P//BM].sort().values
        sub_data, train = self.collator.collate(batch)
        sub_index = sub_data.edge_index

        if check:
            # check results
            sub_src_degree = degree(sub_index[0])
            sub_dst_degree = degree(sub_index[1])
            main_nodes = self.collator.batch_nodes(batch)
            main_size = main_nodes.size(0)
            pivt_remove = self.collator.batch_pivots(batch)
            pivt_mask = torch.ones(self.pivots.num_nodes, dtype=torch.bool)
            pivt_mask[pivt_remove] = False
            pivt_remain = mask_to_index(pivt_mask)
            orig_main_nodes = self.partitions.node_map[main_nodes]
            orig_pivt_nodes = self.pivots.node_map[pivt_remain]
            orig_nodes = torch.cat([orig_main_nodes, orig_pivt_nodes])
            sort_nodes = orig_nodes.sort().values
            assert (sort_nodes[1:] - sort_nodes[:-1] != 0).all(), \
                "some pivot nodes are also main nodes, should remove them"

            baseline_train_mask = torch.zeros((self.baseline[0].num_nodes,), dtype=torch.bool)
            baseline_train_mask[self.baseline.get_idx_split('train')] = True
            orig_train_nodes = orig_main_nodes[baseline_train_mask[orig_main_nodes]]
            baseline_index = self.baseline[0].edge_index
            baseline_sp = SparseTensor(row=baseline_index[1], col=baseline_index[0], is_sorted=True)
            baseline_index = baseline_sp[orig_nodes, orig_nodes]
            baseline_src_deg = degree(baseline_index.coo()[0])
            baseline_dst_deg = degree(baseline_index.coo()[1])
            assert (sub_dst_degree[:main_size] == baseline_dst_deg[:main_size]).all(),\
                "Incorrect in-degrees for some main nodes"
            assert (sub_dst_degree[main_size:][pivt_mask] == baseline_dst_deg[main_size:]).all(),\
                "Incorrect in-degrees for some pivot nodes"
            assert (sub_src_degree[:main_size] == baseline_src_deg[:main_size]).all(),\
                "Incorrect out-degrees for some main nodes"
            # check node data
            orig_nodes = torch.cat([orig_main_nodes, self.pivots.node_map])
            baseline_data = self.baseline[0].x[orig_nodes]
            assert (sub_data.x == baseline_data).all(),\
                "Incorrect node feature"
            baseline_y = self.baseline[0].y[orig_train_nodes]
            assert (sub_data.y[train] == baseline_y).all(),\
                "Incorrect node label"
            # NOTE: out-degrees for some pivot nodes won't be the same
            # but it's ok, because those extra edges are guaranteed not to be sampled
            # assert (sub_src_degree[main_size:][pivt_mask] == baseline_src_deg[main_size:]).all(),\
            #     "Incorrect out-degrees for some pivot nodes"
            logger.info("Edge Index check passed")

        del sub_data, sub_index


if __name__ == "__main__":
    test = unittest.TestSuite()
    test.addTest(TestCollatePivots('test_products'))
    unittest.TextTestRunner().run(test)
    # unittest.main()

'''
Some results on home machine:

### ogbn-products

### partition: 64
rand                 cuts=121785504 imbalance=(1.011, 0.005) time=0.02s 
Fennel-unopt         cuts=46671798 imbalance=(1.061, 0.017) time=1.39s 
Fennel-par           cuts=47360424 imbalance=(1.074, 0.015) time=0.31s 
Fennel-opt           cuts=46764758 imbalance=(1.078, 0.024) time=1.81s 
reFennel-unopt       cuts=34042840 imbalance=(1.099, 0.027) time=4.07s 
reFennel-par         cuts=34189080 imbalance=(1.088, 0.029) time=0.93s 
reFennel-opt         cuts=33312074 imbalance=(1.099, 0.033) time=5.34s 
Fennel-deg-unopt     cuts=26057408 imbalance=(1.0, 0.0) time=1.31s 
Fennel-deg-par       cuts=48494740 imbalance=(1.099, 0.063) time=0.40s 
Fennel-deg-opt       cuts=25785432 imbalance=(1.0, 0.0) time=1.57s 
reFennel-deg-unopt   cuts=21974646 imbalance=(1.086, 0.022) time=4.06s 
reFennel-deg-par     cuts=31370726 imbalance=(1.099, 0.047) time=1.25s 
reFennel-deg-opt     cuts=21683704 imbalance=(1.089, 0.022) time=5.10s 
metis                cuts=16657064 imbalance=(1.03, 0.024) time=20.81s 

### partition: 1024
Data(num_nodes=2449029, x=[2449029, 100], y=[2449029, 1], adj_t=[2449029, 2449029, nnz=123718280])
rand                 cuts=123597434 imbalance=(1.064, 0.02) time=0.02s
Fennel-unopt         cuts=56692032 imbalance=(1.099, 0.043) time=3.60s
Fennel-par           cuts=56979904 imbalance=(1.1, 0.044) time=0.66s
Fennel-opt           cuts=56131804 imbalance=(1.099, 0.043) time=2.17s
reFennel-unopt       cuts=46734414 imbalance=(1.099, 0.05) time=10.31s
reFennel-par         cuts=47163722 imbalance=(1.099, 0.05) time=2.00s
reFennel-opt         cuts=46527934 imbalance=(1.099, 0.049) time=6.22s
Fennel-deg-unopt     cuts=47358296 imbalance=(1.099, 0.004) time=3.36s
Fennel-deg-par       cuts=61708608 imbalance=(1.099, 0.057) time=0.65s
Fennel-deg-opt       cuts=47305714 imbalance=(1.099, 0.004) time=1.81s
reFennel-deg-unopt   cuts=39637970 imbalance=(1.099, 0.025) time=10.14s
reFennel-deg-par     cuts=50454110 imbalance=(1.099, 0.052) time=2.22s
reFennel-deg-opt     cuts=39624648 imbalance=(1.099, 0.026) time=6.02s
metis                cuts=39219308 imbalance=(1.029, 0.028) time=37.60s

### partition: 4096
rand                 cuts=123688242 imbalance=(1.16, 0.04) time=0.02s
Fennel-unopt         cuts=68014640 imbalance=(1.098, 0.05) time=9.94s
Fennel-par           cuts=70060344 imbalance=(1.098, 0.051) time=1.13s
Fennel-opt           cuts=67879310 imbalance=(1.098, 0.05) time=1.99s
reFennel-unopt       cuts=59943776 imbalance=(1.098, 0.055) time=30.79s
reFennel-par         cuts=60792116 imbalance=(1.098, 0.056) time=3.84s
reFennel-opt         cuts=60101560 imbalance=(1.098, 0.056) time=7.62s
Fennel-deg-unopt     cuts=67298390 imbalance=(1.098, 0.004) time=10.43s
Fennel-deg-par       cuts=72441172 imbalance=(1.098, 0.059) time=1.38s
Fennel-deg-opt       cuts=67493824 imbalance=(1.098, 0.004) time=2.34s
reFennel-deg-unopt   cuts=56675088 imbalance=(1.098, 0.025) time=31.71s
reFennel-deg-par     cuts=61889126 imbalance=(1.098, 0.056) time=4.19s
reFennel-deg-opt     cuts=56804620 imbalance=(1.098, 0.025) time=7.00s
metis                cuts=57336176 imbalance=(1.028, 0.027) time=66.81s


### Summary

1. Fennel-par is faster and has similar quality with Fennel-opt

2. Fennel-opt's runtime grows very little from k=1024->4096. It is not the case for
Fennel-unopt, Fennel-par or their deg-ordered variations, or METIS

3. Fennel-deg-opt produces very good partitions even when compared with METIS

4. Fennel-deg-par leads to noticeable worse partitions. This is probably because Fennel-deg-par
tends to put the beginning, high-degree nodes in the stream to the same partition due to delay of
updating partitioning scores. These high-degree nodes are also very influential in deciding the
partitioning of nodes later in the stream. To tackle this issue, We propose __warmup__, which is
to process a small portion (e.g. 1%) of the nodes in the stream sequentially before switching back
to parallel, approximate region. Results with warmup below:

#partition: 64
rand                 cuts=121787630 imbalance=(1.013, 0.005) time=0.02s
Fennel-deg-unopt     cuts=26057408 imbalance=(1.0, 0.0) time=1.26s
Fennel-deg-par       cuts=28990522 imbalance=(1.099, 0.047) time=0.40s
Fennel-deg-opt       cuts=25785432 imbalance=(1.0, 0.0) time=1.53s
reFennel-deg-unopt   cuts=21974646 imbalance=(1.086, 0.022) time=3.89s
reFennel-deg-par     cuts=22954496 imbalance=(1.099, 0.035) time=1.29s
reFennel-deg-opt     cuts=21683704 imbalance=(1.089, 0.022) time=5.02s
metis                cuts=16657064 imbalance=(1.03, 0.024) time=20.81s

#partition: 1024
rand                 cuts=123597840 imbalance=(1.069, 0.021) time=0.02s
Fennel-deg-unopt     cuts=47358296 imbalance=(1.099, 0.004) time=3.76s
Fennel-deg-par       cuts=48789316 imbalance=(1.1, 0.05) time=0.78s
Fennel-deg-opt       cuts=47305714 imbalance=(1.099, 0.004) time=2.11s
reFennel-deg-unopt   cuts=39637970 imbalance=(1.099, 0.025) time=11.30s
reFennel-deg-par     cuts=40505100 imbalance=(1.099, 0.047) time=2.55s
reFennel-deg-opt     cuts=39624648 imbalance=(1.099, 0.026) time=6.90s
metis                cuts=39219308 imbalance=(1.029, 0.028) time=39.26s

#partition: 4096
rand                 cuts=123687912 imbalance=(1.164, 0.041) time=0.02s
Fennel-deg-unopt     cuts=67298390 imbalance=(1.098, 0.004) time=10.58s
Fennel-deg-par       cuts=67109912 imbalance=(1.098, 0.048) time=1.61s
Fennel-deg-opt       cuts=67493824 imbalance=(1.098, 0.004) time=2.14s
reFennel-deg-unopt   cuts=56675088 imbalance=(1.098, 0.025) time=30.11s
reFennel-deg-par     cuts=57997404 imbalance=(1.098, 0.057) time=4.84s
reFennel-deg-opt     cuts=56804620 imbalance=(1.098, 0.025) time=6.71s
metis                cuts=57336176 imbalance=(1.028, 0.027) time=66.81s

4. Fennel-deg-par's partitioning quality gets better with k=4096, since the staleness of scores depends
positively on num_workers and negatively on k.

5. Fennel-par's runtime scaling is better when k becomes larger, possibly because of less data race
on the partition scores.

6. Fennel-opt's runtime is not better than Fennel-unopt when k is very small. The inflection point
seems related to the degree distribution, but k=64/128 works well in practice.

TODO:
-[x]. try omp dynamic parallelism [works]
-[x]. parallelize Fennel-LB-opt. We could try lock-based version for Fennel-LB-opt because of label parallelism [not scaling on ogbn-products, because of uneven distribution of locks]
-[]. (harder) better parallelization of Fennel-LB-opt if locking doesn't work
-[]. make the parameter `slack` a tensor for each label to Fennel-LB variants
-[]. change streaming order to training node first, then degree-ordered/random for the rest.
'''

'''
Evaluations on large datasets: ogbn-papers100M

1. Runtime

Using DGL's shipped METIS with multi-constraints to balance label distributions in each partition (**Note that we are using P=64 here, otherwise METIS will crash**):

```
Converting to homogeneous graph takes 6.891s, peak mem: 190.222 GB                                                                                                                                                                                                               
Convert a graph into a bidirected graph: 152.911 seconds, peak memory: 233.961 GB                                                                                                                                                                                                
Construct multi-constraint weights: 40.540 seconds, peak memory: 500.560 GB        
[18:28:28] /home/tianhaoh/proj/dgl/src/graph/transform/metis_partition_hetero.cc:87: Partition a graph with 111059956 nodes and 3228124712 edges into 64 parts and get 1103610465 edge cuts
Metis partitioning: 4845.827 seconds, peak memory: 599.208 GB   
Assigning nodes to METIS partitions takes 5040.300s, peak mem: 599.208 GB
Reshuffle nodes and edges: 210.651 seconds
Split the graph: 428.715 seconds
```

ReFennel-LB uses little extra memory other than the graph structure itself: **28.4GB**, and finishes the partitioning in 1808s (5 runs) / 1160s (3 runs).

2. Partition quality (edge cuts)

When P=1024,
- DGL gives #cuts = 1529110188
- Fennel-LB (random node order) #cuts = 1322196642 (5 runs)
- Fennel-LB (degree node order) #cuts = 1579096508 (3 runs)

'''

import unittest
import time, os
import torch
from data import partitioner

class TestScatterAppend(unittest.TestCase):
    def test_scatter(self):
        ids = torch.arange(1024*1024)
        psize = 1024
        assigns = partitioner.RandomNodePartitioner(ids, psize).partition()

        parts = partitioner.group(ids, assigns, psize)
        assert len(parts) == psize
        parts_seq = partitioner.scatter_append(
            dim=0, index=assigns, src=ids, return_sequence=True,
        )
        assert len(parts_seq) == psize, f"len(parts_seq)={len(parts_seq)}!={psize}"
        for i, p in enumerate(parts):
            assert (p == parts_seq[i]).all()
            assert (assigns[p] == i).all()
    
    def test_scatter_some_bin_empty(self):
        ids = torch.arange(1024*1024)
        psize = 1024
        assigns = partitioner.RandomNodePartitioner(ids, psize).partition()
        assigns[assigns==1] = 0
        assigns[assigns==3] = 4

        parts = partitioner.group(ids, assigns, psize)
        assert len(parts) == psize
        parts_seq = partitioner.scatter_append(
            dim=0, index=assigns, src=ids, return_sequence=True,
        )
        assert len(parts_seq) == psize, f"len(parts_seq)={len(parts_seq)}!={psize}"
        for i, p in enumerate(parts):
            assert (p == parts_seq[i]).all()
            assert (assigns[p] == i).all()

from pathlib import Path
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask, mask_to_index
from data.partitioner import (
    RandomNodePartitioner,
    MetisNodePartitioner,
    MetisWeightedPartitioner,
    FennelPartitioner,
    FennelPartitionerPar,
    ReFennelPartitioner,
    FennelStrataPartitioner,
    FennelStrataPartitionerPar,
)

# fennel partitioners with node streamed in a degree descending order
class FennelDegOrderPartitioner(FennelPartitioner):
    def __init__(self, g, psize, name='Fennel-deg', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices

class FennelDegOrderPartitionerPar(FennelPartitionerPar):
    def __init__(self, g, psize, name='Fennel-deg-par', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices

class FennelStrataDegOrderPartitioner(FennelStrataPartitioner):
    def __init__(self, g, psize, name='Fennel-strata-deg', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices

class DglMetisStrataPartitioner(partitioner.NodePartitioner):
    def __init__(self, name, root, psize, labels):
        from ogb.nodeproppred import DglNodePropPredDataset
        dataset = DglNodePropPredDataset(name, root)
        data = dataset[0]
        g, labels = data
        labels = labels.flatten().int()
        num_labels = labels.max().item() + 1
        train_nid = dataset.get_index_split()['train']
        train_mask = index_to_mask(train_nid, g.num_nodes())
        labels[~train_mask] = num_labels
        self.root = root
        self.labels = labels
        super().__init__(g, psize, name=f'DGL-Metis-LB-{name}')
    
    def partition(self):
        import dgl
        dgl.distributed.partition.graph_partition(
            self.g, self.name, self.psize, self.root,
            part_method='metis', balance_ntypes=self.labels
        )

class TestPartitioner(unittest.TestCase):
    def setUp(self):
        super().setUp()
        path = Path('/opt/datasets')
        dataset = PygNodePropPredDataset(
            root=path, name='ogbn-products',
            # pre_transform=T.ToUndirected(),
            transform=T.ToSparseTensor()
        )
        self.data = dataset[0]
        split = dataset.get_idx_split()
        self.train_nid, self.val_nid, self.test_nid = split['train'], split['valid'], split['test']
        self.labels = self.data.y.flatten()
        self.psize = int(os.environ.get('psize', 64))
        print()
        print(dataset[0])
        print("#train:", self.train_nid.size(0), "#partition:", self.psize)

    def edge_cuts(self, pt_assigns):
        src, dst, _ = self.data.adj_t.coo()
        return (pt_assigns[src]-pt_assigns[dst] != 0).int().sum().item()

    def load_ratio(self, assigns, nids=None):
        nid_assigns = assigns[nids] if nids is not None else assigns
        nid_size = nid_assigns.size(0)
        if nid_size == 0:
            return float('nan'), 0
        loads = torch.histc(nid_assigns.float(), bins=self.psize, max=self.psize) \
            / (nid_size / self.psize)
        return int(loads.max().item()*1000)/1000, int(loads.std().item()*1000)/1000

    def run_partition(self, p):
        tic = time.time()
        assigns = p.partition()
        toc = time.time()
        return assigns, toc - tic
    
    def test_partition_balance_nodes(self):
        data, psize = self.data, self.psize
        partitioners = [
            RandomNodePartitioner(data, psize),
            FennelPartitioner(data, psize, use_opt=False, name="Fennel-unopt"),
            FennelPartitionerPar(data, psize),
            FennelPartitioner(data, psize, use_opt=True, name="Fennel-opt"),
            ReFennelPartitioner(
                data, psize, runs=5,
                base=FennelPartitioner,
                use_opt=False,
                name="reFennel-unopt"
            ),
            ReFennelPartitioner(data, psize, runs=5, base=FennelPartitionerPar, name="reFennel-par"),
            ReFennelPartitioner(data, psize, runs=5, base=FennelPartitioner, name="reFennel-opt"),
            # deg-ordered version
            FennelDegOrderPartitioner(data, psize, use_opt=False, name="Fennel-deg-unopt"),
            FennelDegOrderPartitionerPar(data, psize),
            FennelDegOrderPartitioner(data, psize, use_opt=True, name="Fennel-deg-opt"),
            ReFennelPartitioner(
                data, psize, runs=5,
                base=FennelDegOrderPartitioner,
                use_opt=False,
                name="reFennel-deg-unopt"
            ),
            ReFennelPartitioner(
                data, psize, runs=5,
                base=FennelDegOrderPartitionerPar,
                init='rand',
                name="reFennel-deg-par"
            ),
            ReFennelPartitioner(
                data, psize, runs=5,
                base=FennelDegOrderPartitioner,
                name="reFennel-deg-opt"
            ),
            # metis
            MetisNodePartitioner(data, psize),
        ]
        for p in partitioners:
            assigns, runtime = self.run_partition(p)
            valid = (assigns < self.psize).sum().item() == self.data.size(0)
            valid = valid and (assigns >= 0).sum().item() == self.data.size(0)
            if not valid:
                print(f"{p.name} produced invalid partitions!")
            print(
                f"{p.name:<20} cuts={self.edge_cuts(assigns)} "
                f"imbalance={self.load_ratio(assigns)} "
                f"time={runtime:.2f}s" 
            )
    
    def test_partition_balance_train(self):
        data, psize = self.data, self.psize
        train_mask = index_to_mask(self.train_nid, data.size(0))
        train_labels = train_mask.clone().int()
        train_labels[~train_mask] = -1
        partitioners = [
            RandomNodePartitioner(data, psize),
            # stratified fennel
            FennelStrataDegOrderPartitioner(data, psize, labels=train_mask),
            ReFennelPartitioner(
                data, psize, runs=5,
                base=FennelStrataDegOrderPartitioner,
                labels=train_labels,
                name="reFennel-strata"
            ),
            # metis
            MetisNodePartitioner(data, psize),
            MetisWeightedPartitioner(data, psize, node_weights=train_mask),
        ]
        for p in partitioners:
            assigns, runtime = self.run_partition(p)
            assert (assigns < self.psize).sum().item() == self.data.size(0)
            assert (assigns >= 0).sum().item() == self.data.size(0)
            print(
                f"{p.name:<20} cuts={self.edge_cuts(assigns)} "
                f"imbalance(train)={self.load_ratio(assigns, train_mask)} "
                f"imbalance(non-train)={self.load_ratio(assigns, ~train_mask)} "
                f"time={runtime:.2f}s"
            )
    
    def test_partition_balance_labels(self):
        data, psize = self.data, self.psize
        train_mask = index_to_mask(self.train_nid, data.size(0))
        train_labels = self.labels.clone().int()
        num_labels = self.labels.max().item() + 1
        train_labels[~train_mask] = num_labels
        nid_per_label = [
            mask_to_index(train_labels==l) for l in range(num_labels)
        ]
        print([round(len(nid)/psize,2) for nid in nid_per_label])
        partitioners = [
            RandomNodePartitioner(data, psize),
            # stratified fennel
            FennelStrataPartitioner(
                data, psize, slack=2,
                labels=train_labels,
                name="Fennel-strata",
            ),
            ReFennelPartitioner(
                data, psize, slack=2, beta=1, runs=5,
                base=FennelStrataPartitioner,
                labels=train_labels,
                name="reFennel-strata",
            ),
            FennelStrataDegOrderPartitioner(
                data, psize, slack=2,
                labels=train_labels,
            ),
            ReFennelPartitioner(
                data, psize, slack=2, beta=1, runs=5,
                base=FennelStrataDegOrderPartitioner,
                labels=train_labels,
                name="reFennel-strata-deg",
            ),
            FennelStrataPartitionerPar(
                data, psize, slack=2,
                labels=train_labels,
                name="Fennel-strata-par",
            ),
            ReFennelPartitioner(
                data, psize, slack=2, beta=1, runs=5,
                base=FennelStrataPartitionerPar,
                labels=train_labels,
                name="reFennel-strata-par",
            ),
            # metis
            MetisNodePartitioner(data, psize),
            MetisWeightedPartitioner(data, psize, node_weights=train_mask),
        ]
        for p in partitioners:
            assigns, runtime = self.run_partition(p)
            assert (assigns < self.psize).sum().item() == self.data.size(0)
            assert (assigns >= 0).sum().item() == self.data.size(0)
            imbalance = [self.load_ratio(assigns, nid) for nid in nid_per_label]
            print(
                f"{p.name:<20} cuts={self.edge_cuts(assigns)}, "
                f"imbalance(train)={self.load_ratio(assigns, train_mask)} "
                f"imbalance(non-train)={self.load_ratio(assigns, ~train_mask)} "
                f"imbalance/label=\n{imbalance}, "
                f"time={runtime:.2f}s"
            )

if __name__ == "__main__":
    # test = unittest.TestSuite()
    # test.addTest(TestPartitioner('test_partition_balance_labels'))
    # unittest.TextTestRunner().run(test)
    unittest.main()


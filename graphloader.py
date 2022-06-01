import os
import os.path as osp
from enum import Enum
from functools import namedtuple
import torch
import dgl

import utils
import datasets, gnnos

__all__ = [
    'PartitionMethod',
    'GraphLoader',
    'PartitionSampler',
    'PartitionedGraphLoader',
    'HBatchSampler',
    'HBatchGraphLoader']

GraphLoaderArgs = namedtuple("args", ["name", "root", "mmap"])
def _args(**kwargs):
    default_args = {'mmap': False, 'root': '.'}
    return GraphLoaderArgs(**{**default_args, **kwargs})

def external_partition(graph, psize):
    '''
    Partition graph from external assignments
    '''
    graph_path = graph.metadata[1].path
    data_dir = osp.join(graph_path, f"partitions/EXTERNAL/p{psize}")
    assigns = torch.load(osp.join(data_dir, f"p{psize}.pt")).int()
    return gnnos.node_partitions(psize, assigns)

class PartitionMethod(Enum):
    RANDOM = 0
    METIS = 1
    HBATCH = 2
    EXTERNAL = 3

    @staticmethod
    def fn(p_method):
        fn_list = [
            gnnos.random_partition,
            dgl.metis_partition_assignment,
            gnnos.good_partition,
            external_partition
        ]
        return fn_list[p_method.value]

class GraphLoader(object):
    '''
    GraphLoader loads the stored graph dataset from the given folder.
    It could provide an in-memory version of the dataset or an mmap view
    - graph topology: in-memory
    - node features: in-memory (mmap=False) or on-storage (mmap=True)
    - paritioned graph: no
    '''
    def __init__(self, **kwargs):
        args = _args(**kwargs)
        self.name = args.name
        self.canonical_name = self.name.replace('-', '_')
        self.root = args.root
        self.mmap = args.mmap
        self.dataset_dir = osp.join(self.root, self.canonical_name)

        graphs, _ = dgl.load_graphs(osp.join(self.dataset_dir, "graph.dgl"))
        self.graph = graphs[0]
        for k, v in self.graph.ndata.items():
            if k.endswith('_mask'):
                self.graph.ndata[k] = v.bool()

        feat_shape_file = osp.join(self.dataset_dir, "feat.shape")
        feat_file = osp.join(self.dataset_dir, "feat.feat")
        shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64', shape=(2,)))
        if self.mmap:
            self.node_features = utils.memmap(feat_file, random=True, mode='r', dtype='float32',
                shape=shape)
        else:
            feat_size  = torch.prod(torch.tensor(shape, dtype=torch.long)).item()
            self.node_features = torch.from_file(feat_file, size=feat_size,
                dtype=torch.float32).reshape(shape)

    def formats(self, formats):
        self.graph = self.graph.formats(formats)

    def feature_dim(self):
        return self.node_features.shape[1:]

    def features(self, indices) -> torch.Tensor:
        tensor = self.node_features[indices]
        if self.mmap:
            # where most accesses to the storage happens
            # torch.from_numpy returns zerocopy view of underlying DiskTensor
            # PyTorch would complain about DiskTensor being not writable
            # If there's any write to the tensor, the program would segfault
            return torch.from_numpy(tensor)
        else:
            return tensor

    def labels(self):
        return self.graph.ndata['label']

    def num_classes(self):
        return torch.max(self.graph.ndata['label']).item()

    @staticmethod
    def _nonzero_idx(tensor_1d):
        return torch.nonzero(tensor_1d, as_tuple=True)[0]

    def train_idx(self):
        return GraphLoader._nonzero_idx(self.graph['train_mask'])

    def valid_idx(self):
        return GraphLoader._nonzero_idx(self.graph['valid_mask'])

    def test_idx(self):
        return GraphLoader._nonzero_idx(self.graph['test_mask'])

    def get_idx_split(self):
        return self.train_idx(), self.valid_idx(), self.test_idx()


def partition_offsets(partitions):
    return torch.cumsum(
        torch.tensor([0] + [len(part) for part in partitions], dtype=torch.long), dim=0)

def split_tensor(tensor, intervals):
    return [tensor[intervals[i]:intervals[i+1]] for i in range(len(intervals)-1)]


class PartitionSampler(dgl.dataloading.Sampler):
    def __init__(self, partitions, device='cpu', prefetch_ndata=None, prefetch_edata=None):
        super().__init__()
        self.partitions = partitions
        self.device = device
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []

    def sample(self, g, partition_ids):
        '''
        sample partitions and generate subgraphs without features attached
        '''
        partitions = [self.partitions[i] for i in partition_ids]
        node_ids = torch.cat(partitions)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.device)
        dgl.dataloading.set_node_lazy_features(sg, self.prefetch_ndata)
        dgl.dataloading.set_edge_lazy_features(sg, self.prefetch_edata)

        intervals = partition_offsets(partitions)
        return intervals, partition_ids, sg



class PartitionedGraphLoader(GraphLoader):
    '''
    PartitionedGraphLoader loads a partitioned graph dataset
    under the "partitions" folder.
    If the graph is not yet partitioned, it generates a new partitioning,
    reshuffles the node features if needed and cache the results
    under the "partitions" folder.
    - graph topology: in-memory
    - node features: on-storage
    - paritioned graph: yes
    '''
    def __init__(self, p_size, p_method=PartitionMethod(0), overwrite=False, **kwargs):
        super(PartitionedGraphLoader, self).__init__(**kwargs)
        self.p_size = p_size
        self.p_method = p_method
        p_func = PartitionMethod.fn(p_method)

        partition_dir = osp.join(self.dataset_dir, f"partitions/{self.p_method.name}")
        os.makedirs(partition_dir, exist_ok=True)
        partition_file = osp.join(partition_dir, f"p{p_size}.pt")
        partition_features_file = osp.join(partition_dir, f"features{p_size}")
        partition_features_shape = osp.join(partition_dir, "features.shape")
        if not overwrite:
            try:
                self.partitions = torch.load(partition_file)
                if self.mmap:
                    shape = tuple(utils.memmap(partition_features_shape, mode='r', dtype='int64'))
                    self.shuffled_features = utils.memmap(partition_features_file, mode='r',
                        dtype='float32', shape=shape)
            except Exception as e:
                print(f"[{type(self).__name__}] Exception thrown when loading "
                    f"cached {self.p_method.name}-partitioning results:\n\t{e}")
                overwrite = True

        if overwrite:
            # generate new partitions and shuffled features
            print(f"[{type(self).__name__}] Partition {self.name} with {self.p_method.name}")
            self.partitions = p_func(self.graph, self.p_size,
                mask=self.graph.ndata['train_mask'].int())
            torch.save(self.partitions, partition_file)
            if self.mmap:
                shuffled_features = PartitionedGraphLoader.shuffle(self.node_features,
                    self.partitions, partition_features_file)
                shape_mmap = utils.memmap(partition_features_shape, mode='w+', dtype='int64',
                    shape=(len(shuffled_features.shape),))
                shape_mmap[:] = shuffled_features.shape
                shape_mmap.flush()
                self.shuffled_features = shuffled_features

        self.offsets = partition_offsets(self.partitions)

    @staticmethod
    def shuffle(features, partition_list, new_file):
        '''
        given a partition list, input node features, generate a shuffled feature file
        where node features in the same partition are put adjacent,
        '''
        new_features = utils.memmap(new_file, mode='w+', dtype='float32', shape=features.shape)
        offset = 0
        for partition in partition_list:
            partition_len = len(partition)
            # dgl.subgraph(nodes) keeps the order of nodes in the returned subgraph
            new_features[offset:offset+partition_len, :] = features[partition, :]
            offset += partition_len

        new_features.flush()
        # prevent any future writes on the mmap file
        new_features = utils.memmap(new_file, mode='r', dtype='float32', shape=features.shape)
        return new_features

    '''
    def check_shuffle(self):
        assert self.mmap, "Features are not shuffled if mmap is False"
        for i, (nodes, features) in enumerate(self.pg):
            nodes = torch.cat(nodes)
            features = torch.cat(features)
            assert (features == self.get_features(nodes)).all(), \
                f"Inconsistent results for Partition {i}"
    '''

    def partition_idx(self):
        return torch.arange(0, self.p_size)

    def feat_partition(self, pid):
        '''
        Get the feature tensor of the pid-th partition
        '''
        if self.mmap:
            return torch.from_numpy(self.shuffled_features[self.offsets[pid]:self.offsets[pid+1]])
        else:
            return self.features(self.partitions[pid])

    def gather_feat_partitions(self, indices):
        feats = [self.feat_partition(i) for i in indices]
        return torch.cat(feats)


class HBatchGraphLoader:
    '''
    Load partitioned graph and features backed by gnnos.TensorStore
    - graph topology: on-storage
    - node features: on-storage
    - paritioned graph: yes
    The only data in the CPU memory are train/val/test masks
    '''
    def __init__(self, name: str, root: str, p_size, p_method=PartitionMethod(0)):
        self.name = name
        self.canonical_name = self.name.replace('-', '_')
        self.root = root
        self.dataset_dir = osp.join(root, self.canonical_name)
        self.p_size = p_size
        self.p_method = p_method

        if name == "oag-paper":
            data = datasets.load_oag(self.dataset_dir)
        elif name == "mag240m":
            data = datasets.load_mag240m(self.dataset_dir)
        elif name.startswith("ogbn"):
            data = datasets.load_ogb(self.dataset_dir)
        else:
            raise ValueError("Unknown dataset, please choose from oag-paper, mag240m, or OGB")
        # unpack loaded data
        graph, node_features, self.is_multilabel, labels, self.masks = data

        print("Loaded dataset")

        # try loading cached partition files: assignments, BCOO, features, labels
        create = False
        data_dir = osp.join(self.dataset_dir, f'partitions/{p_method.name}/p{p_size}')
        try:
            pg, shuffled_features, shuffled_labels, partitions = \
                datasets.load_partitions(data_dir)
        except Exception as e:
            print(e)
            create = True
        if create:
            os.makedirs(data_dir, exist_ok=True)
            partitions = PartitionMethod.fn(self.p_method)(graph, self.p_size)
            pg, shuffled_features, shuffled_labels, _ = datasets.create_partitions(
                graph, node_features, labels, partitions, data_dir)

        self.graph = pg
        self.partitions = partitions
        self.shuffled_features = shuffled_features
        self.shuffled_labels = shuffled_labels

    def feature_dim(self):
        return self.shuffled_features.metadata.shape[1:]

    def num_classes(self):
        if self.is_multilabel:
            return self.labels.metadata.shape[1]
        else:
            labels: torch.Tensor = self.shuffled_labels.tensor()
            return torch.max(labels[~labels.isnan()]).long().item() + 1

    def partition_idx(self):
        return torch.arange(0, self.p_size)

    def gather_node_partitions(self, indices):
        nodes = [self.partitions[i] for i in indices]
        return torch.cat(nodes)

    def gather_feat_partitions(self, indices):
        slices = [(self.partitions.pos(idx), self.partitions.pos(idx+1)) for idx in indices]
        return gnnos.gather_slices(self.shuffled_features, slices)

    def gather_label_partitions(self, indices):
        slices = [(self.partitions.pos(idx), self.partitions.pos(idx+1)) for idx in indices]
        return gnnos.gather_slices(self.shuffled_labels, slices).long()


if __name__ == "__main__":
    import gc, resource
    def using(point=""):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return '%s: mem=%s MB' % (point, usage[2]/1024.0 )

    gnnos.verbose()

    print(using('before'))
    gloader = HBatchGraphLoader(
        name="ogbn-products", root="/mnt/md0/inputs", p_size=16)
    print(using('after'))
    parts = torch.randint(16, (4,))
    print(f"Fetch partitions: {parts}")
    feats = gloader.gather_feat_partitions(parts)
    nodes = gloader.gather_node_partitions(parts)
    print(feats)
    print(using('tensor'))
    del gloader
    gc.collect()
    print(using('collect'))

    print(using('before'))
    gloader = PartitionedGraphLoader(
        name="ogbn-products", root="/mnt/md0/graphs", p_size=16)
    print(using('after'))
    feats_ref = gloader.features(nodes)
    assert (feats == feats_ref).all()
    print(feats_ref)
    print(using('tensor'))
    del gloader
    gc.collect()
    print(using('collect'))


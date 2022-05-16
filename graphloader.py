import os
import os.path as osp
from enum import Enum
from functools import namedtuple
import torch
import dgl

import utils
import partition_utils as p_utils

__all__ = ['PartitionMethod', 'GraphLoader', 'GraphPartitions', 'PartitionedGraphLoader']

GraphLoaderArgs = namedtuple("args", ["name", "root", "mmap"])
def _args(**kwargs):
    default_args = {'mmap': False, 'root': '.'}
    return GraphLoaderArgs(**{**default_args, **kwargs})

class PartitionMethod(Enum):
    RANDOM = 0
    METIS = 1
    NEV = 2
    EXTERNAL = 3

    @staticmethod
    def select(p_method):
        fn_list = [
            p_utils.get_rand_partition_list,
            p_utils.get_partition_list,
            # TODO: change to get_nev_partition_list
            p_utils.get_rand_partition_list_clusterseed,
            None,
        ]
        return fn_list[p_method.value]

class GraphLoader(object):
    '''
    GraphLoader loads the stored graph dataset from the given folder.
    It could provide an in-memory version of the dataset or an mmap view
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

        feat_shape_file = osp.join(self.dataset_dir, "features.shape")
        feat_file = osp.join(self.dataset_dir, "features")
        # TODO: put (feat_name, feat_shape) in a json or other file, rather than a binary file
        shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64'))
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
    def __init__(self, partitions,  prefetch_ndata=None, prefetch_edata=None):
        super().__init__()
        self.partitions = partitions
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []

    def sample(self, g, partition_ids):
        # sample partitions and generate subgraphs without features attached
        partitions = [self.partitions[i] for i in partition_ids]
        node_ids = torch.cat(partitions)
        sg = g.subgraph(node_ids, relabel_nodes=True)
        dgl.dataloading.set_node_lazy_features(sg, self.prefetch_ndata)
        dgl.dataloading.set_edge_lazy_features(sg, self.prefetch_edata)

        intervals = partition_offsets(partitions)
        # sg_nodes = sg.nodes()
        # sg_partitions = [sg_nodes[intervals[i]:intervals[i+1]] for i in range(len(partition_ids))]
        return sg, intervals, partition_ids



class PartitionedGraphLoader(GraphLoader):
    '''
    PartitionedGraphLoader loads a partitioned graph dataset
    under the "partitions" folder.
    If the graph is not yet partitioned, it generates a new partitioning,
    reshuffles the node features if needed and cache the results
    under the "partitions" folder.
    '''
    def __init__(self, p_size, p_method=PartitionMethod(0), overwrite=False, **kwargs):
        super(PartitionedGraphLoader, self).__init__(**kwargs)
        self.p_size = p_size
        self.p_method = p_method
        p_func = PartitionMethod.select(p_method)

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

        # # creates a map of nid -> offsets, sorted by nid
        # feat_offsets = torch.arange(0, features.shape[0])
        # par_array = torch.cat(partition_list)
        # nid_to_feat = torch.stack((par_array, feat_offsets))
        # nid_to_feat = nid_to_feat[:, nid_to_feat[0].argsort()]
        # new_features[nid_to_feat[1], :] = features[:, :]

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

    def partition_features(self, pid):
        '''
        Get the feature tensor of the pid-th partition
        '''
        if self.mmap:
            return torch.from_numpy(self.shuffled_features[self.offsets[pid]:self.offsets[pid+1]])
        else:
            return self.features(self.partitions[pid])

if __name__ == "__main__":
    import gc, resource
    def using(point=""):
        usage=resource.getrusage(resource.RUSAGE_SELF)
        return '%s: mem=%s MB' % (point, usage[2]/1024.0 )
    print(using('before'))
    gloader = GraphLoader(name="ogbn-products", root="/mnt/md0/graphs", mmap=True)
    print(using('after'))
    gloader.formats(['csc'])
    print(using('formats'))
    gc.collect()
    print(using('collect'))
    gloader.graph.edges()
    print(gloader.graph.formats())
    print(using('last'))

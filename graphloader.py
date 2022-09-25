from dataclasses import dataclass
import os, json
import os.path as osp
from enum import Enum
from functools import namedtuple
from typing import Union
import numpy as np
import torch, dgl

import sampler, utils
import gnnos

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

def load_mag240m_citation(root, mmap=False):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root=root)
    num_classes = dataset.num_classes
    ei_cites = dataset.edge_index('paper', 'cites', 'paper')
    graph = dgl.graph((np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]])), num_nodes=dataset.num_papers)
    print(graph)
    graph.ndata["label"] = torch.tensor(dataset.all_paper_label)

    train_idx = dataset.get_idx_split('train')
    val_idx = dataset.get_idx_split('valid')
    test_idx = dataset.get_idx_split('valid')

    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_idx] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['valid_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    paper_feat = dataset.paper_feat if mmap else dataset.all_paper_feat
    del dataset
    return graph, num_classes, paper_feat

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

        if "mag240m" in self.name:
            self.graph, num_classes, feats = load_mag240m_citation(self.root, self.mmap)
            if not self.mmap:
                self.node_features = torch.from_numpy(feats)
                self.node_features.share_memory_()
        else:
            graphs, _ = dgl.load_graphs(osp.join(self.dataset_dir, "graph.dgl"))
            self.graph = graphs[0]

            feat_shape_file = osp.join(self.dataset_dir, "feat.shape")
            feat_file = osp.join(self.dataset_dir, "feat.feat")
            shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64', shape=(2,)))
            if self.mmap:
                self.node_features = utils.memmap(feat_file, random=True, mode='r', dtype='float32',
                    shape=shape)
            else:
                feat_size  = torch.prod(torch.tensor(shape, dtype=torch.long)).item()
                self.node_features = torch.from_file(
                    feat_file, size=feat_size, dtype=torch.float32).reshape(shape)
                self.node_features.share_memory_()

        for k, v in self.graph.ndata.items():
            if k.endswith('_mask'):
                self.graph.ndata[k] = v.bool()

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
        labels = self.labels()
        return torch.max(labels[~labels.isnan()]).long().item() + 1

    @staticmethod
    def _nonzero_idx(tensor_1d):
        return torch.nonzero(tensor_1d, as_tuple=True)[0]

    def train_idx(self):
        return GraphLoader._nonzero_idx(self.graph.ndata['train_mask'])

    def valid_idx(self):
        return GraphLoader._nonzero_idx(self.graph.ndata['valid_mask'])

    def test_idx(self):
        return GraphLoader._nonzero_idx(self.graph.ndata['test_mask'])

    def get_idx_split(self):
        return self.train_idx(), self.valid_idx(), self.test_idx()

# XXX: not using the latest impl of mmap torch.Tensor
class BaselineNodePropPredDataset(object):
    '''
    Baseline, mmap-based dataset
    Loosely follows the interface of NodePropPredDataset in ogb.
    Requires the following inputs:
        dataset dir contains metadata.json;
        metadata.json gives input data locations:
        - graph
        - label
        - node_feat
        - edge_feat
        - train/val/test idx
        extra attributes:
        - num_nodes, num_tasks, task_type, num_classes, is_hetero
    '''
    def __init__(self, name, root = 'dataset', mmap_feat=False, meta_dict = None):
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))
            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            with open(osp.join(self.root, 'metadata.json')) as f_meta:
                self.meta_info = json.load(f_meta)
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        self.num_tasks = int(self.meta_info['num_tasks'])
        self.task_type = self.meta_info['task_type']
        self.num_classes = self.meta_info['num_classes']
        self.is_hetero = self.meta_info['is_hetero']
        self.mmap_feat = mmap_feat

        super(BaselineNodePropPredDataset, self).__init__()
        self.load_data()

    def tensor_from_dict(self, dict, inmem=True, random=False):
        full_path = osp.join(self.root, dict['path'])
        shape = dict['shape']
        size = torch.prod(torch.LongTensor(shape)).item()
        if inmem:
            array = np.fromfile(full_path, dtype=dict['dtype'], count=size)
            return torch.from_numpy(array).reshape(shape)
        else:
            dtype = utils.torch_dtype(dict['dtype'])
            # shared=False to disable modification of tensors
            tensor = torch.from_file(full_path, size=size, dtype=dtype, shared=False).reshape(shape)
            if random:
                utils.madvise_random(tensor.data_ptr(), tensor.numel()*tensor.element_size())
            return tensor

    def load_graph(self, graph_dict):
        if graph_dict['format'] == 'coo':
            edge_index = self.tensor_from_dict(graph_dict['edge_index'], inmem=not self.mmap_feat)
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            utils.using("creating dgl graph (coo)")
            return dgl.graph(data=(src_nodes, dst_nodes),
                num_nodes=self.num_nodes, device='cpu')
        elif graph_dict['format'] in ('csc', 'csr'):
            row_ptr = self.tensor_from_dict(graph_dict['row_ptr'], inmem=not self.mmap_feat)
            col_idx = self.tensor_from_dict(graph_dict['col_idx'], inmem=not self.mmap_feat)
            # disable edge id for now
            #  if 'edge_ids' in graph_dict:
            #      edge_ids = self.tensor_from_dict(graph_dict['edge_ids'], inmem=not self.mmap_feat)
            #  else:
            #      edge_ids = torch.LongTensor()
            utils.using("creating dgl graph (csc/csr)")
            return dgl.graph(data=('csc', (row_ptr, col_idx, edge_ids)))
        else:
            raise NotImplementedError("Only support coo,csc,csr graph formats")

    def load_labels(self):
        return self.tensor_from_dict(self.meta_info['labels'], inmem=not self.mmap_feat, random=True)

    def load_node_feat(self):
        return self.tensor_from_dict(self.meta_info['node_feat'], inmem=not self.mmap_feat, random=True)

    def load_data(self):
        self.num_nodes = self.meta_info['num_nodes']
        self.graph = self.load_graph(self.meta_info['graph'])
        utils.using("graph loaded")
        self.labels = self.load_labels()
        self.node_feat = self.load_node_feat()
        if self.meta_info['edge_feat'] is not None:
            self.edge_feat = self.tensor_from_dict(self.meta_info['edge_feat'],
                inmem=not self.mmap_feat)
        if self.is_hetero:
            self.edge_type = self.tensor_from_dict(self.meta_info['edge_type'],
                inmem=not self.mmap_feat)
        utils.using("dataset loaded")

    def get_idx_split(self):
        idx_dict = self.meta_info['idx']
        train_idx = self.tensor_from_dict(idx_dict['train'])
        valid_idx = self.tensor_from_dict(idx_dict['valid'])
        test_idx = self.tensor_from_dict(idx_dict['test'])
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.labels

    def __len__(self):
        return 1


import tqdm
from datasets import tensor_serialize

@dataclass
class GnnosPartGraph:
    '''
    Partitioned graph in GNNoS
    '''
    num_nodes: int
    psize: int
    src_part_ptr: Union[torch.Tensor, gnnos.TensorStore]    # P + 1
    src_nids: Union[torch.Tensor, gnnos.TensorStore]        # |V|
    dst_ptr: Union[torch.Tensor, gnnos.TensorStore]         # |V| + 1
    dst_nids: Union[torch.Tensor, gnnos.TensorStore]        # }E|

    def adj(self, node_i):
        start, end = self.dst_ptr[node_i], self.dst_ptr[node_i+1]
        adj = self.dst_nids[start:end]
        return self.src_nids[node_i], adj

    def part(self, part_i):
        start, end = self.src_part_ptr[part_i], self.src_part_ptr[part_i+1]
        nodes = self.src_nids[start:end]
        return nodes

    def size(self, part_i) -> int:
        return (self.src_part_ptr[part_i+1] - self.src_part_ptr[part_i]).item()

@dataclass
class GnnosScache:
    num_nodes: int
    psize: int
    # for subgraph induced by src_nids (CSF)
    src_nids: Union[torch.Tensor, gnnos.TensorStore]
    dst_ptr: Union[torch.Tensor, gnnos.TensorStore]
    dst_nids: Union[torch.Tensor, gnnos.TensorStore]
    # for all other edges (COO)
    part_ptr: Union[torch.Tensor, gnnos.TensorStore]
    coo_src: Union[torch.Tensor, gnnos.TensorStore]
    coo_dst: Union[torch.Tensor, gnnos.TensorStore]


def coo_to_csf(edge_index):
    '''
    convert an edge index in coo into the compressed format
    '''
    # sort then stable sort
    edge_np = edge_index.numpy()
    idx = np.argsort(edge_np[1])
    edge_np = edge_np[:, idx] # create a copy
    idx = np.argsort(edge_np[0], kind='stable')
    edge_np[:] = edge_np[:, idx]
    edge_tensor = torch.from_numpy(edge_np)

    # compress source
    nids, counts = torch.unique_consecutive(edge_tensor[0], return_counts=True)
    dst_ptr = torch.cumsum(counts, dim=0)
    return nids, dst_ptr, edge_tensor[1]


def split_graph(edge_index, assignments: torch.Tensor, psize: int, serialize=False, data_dir=None):
    '''
    split the given graph into 1D partitions, based on the node assignments

    returns a tuple of tensors (or a tuple of gnnos tensor info):
    part_ptr: offset array into src_nids for each partitions
    src_nids: source nids
    dst_ptr: offset array int dst_nids for each source node
    dst_nids: concatenated adjacency lists (sorted) of each source nid
    '''
    # get num_node per partition
    num_nodes = len(assignments)

    # group edges by partition assignments
    edge_assigns = assignments.gather(dim=0, index=edge_index[0])
    sorted_assigns, reverse_map = torch.sort(edge_assigns)
    grouped_index = edge_index[:, reverse_map]
    # get number of edges per partition
    _, group_counts = torch.unique_consecutive(sorted_assigns, return_counts=True)
    part_boundary = torch.empty((psize+1,), dtype=group_counts.dtype)
    part_boundary[0] = 0
    torch.cumsum(group_counts, dim=0, out=part_boundary[1:])
    assert part_boundary[-1].item() == len(edge_index[0])

    # accumulate csf from each partition into (part_ptr, src_nids, dst_ptr, dst_nids)
    part_ptr = torch.zeros((psize+1,), dtype=torch.long)
    src_nids = torch.zeros((num_nodes,), dtype=torch.long)
    dst_ptr = torch.zeros((num_nodes+1,), dtype=torch.long)
    dst_nids = torch.zeros((len(edge_index[0]),), dtype=torch.long)
    src_nids_idx, dst_ptr_idx, dst_nids_idx = 0, 1, 0
    dst_ptr_off = 0
    for i in tqdm.tqdm(range(psize)):
        p_src, p_dst_ptr, p_dst_nids = coo_to_csf(grouped_index[:, part_boundary[i]:part_boundary[i+1]])
        # print("Partition", i)
        # check_graph(num_nodes, edge_index, p_src, [0] + list(p_dst_ptr), p_dst_nids)
        pn, dn = len(p_src), len(p_dst_nids)
        part_ptr[i+1] = src_nids_idx + pn
        src_nids[src_nids_idx : src_nids_idx+pn] = p_src
        dst_ptr[dst_ptr_idx : dst_ptr_idx+pn] = p_dst_ptr + dst_ptr_off
        dst_nids[dst_nids_idx : dst_nids_idx+dn] = p_dst_nids
        src_nids_idx += pn
        dst_ptr_idx += pn
        dst_nids_idx += dn
        dst_ptr_off = p_dst_ptr[-1] + dst_ptr_off

    partitioned = GnnosPartGraph(num_nodes, psize, part_ptr, src_nids, dst_ptr, dst_nids)

    if serialize:
        partition_dict = {}
        partition_dict['part_ptr'] = tensor_serialize(part_ptr.numpy(), osp.join(data_dir, 'part_ptr'))
        partition_dict['src_nids'] = tensor_serialize(src_nids[:src_nids_idx].numpy(), osp.join(data_dir, 'src_nids'))
        partition_dict['dst_ptr'] = tensor_serialize(dst_ptr[:dst_ptr_idx].numpy(), osp.join(data_dir, 'dst_ptr'))
        partition_dict['dst_nids'] = tensor_serialize(dst_nids[:dst_nids_idx].numpy(), osp.join(data_dir, 'dst_nids'))
        return partition_dict, partitioned
    else:
        return part_ptr, partitioned

def check_partition(assignments, pg: GnnosPartGraph):
    print("Checking partitions")
    for i in tqdm.tqdm(range(pg.psize)):
        nids = pg.part(i)
        assert (assignments[nids] == i).all(), f"Part {i}"
    print("Passed")

def check_graph(g: dgl.DGLGraph, pg: GnnosPartGraph):
    print("Checking adjs")
    assert g.num_edges() == len(pg.dst_nids)
    for i, _ in enumerate(tqdm.tqdm(pg.src_nids)):
        nid, adj = pg.adj(i) # out edges
        adj_ref, _ = torch.sort(g.in_edges(nid)[0])
        assert (len(adj) == len(adj_ref)) and (adj == adj_ref).all(), \
            f"Node {i}: id={nid}\n{adj}\n{adj_ref}"
    print("Passed")

def check_feat(feat_in_order: torch.Tensor, pg: GnnosPartGraph, feat_shuffled: torch.Tensor):
    print("Checking node feat")
    offset = 0
    for i in tqdm.tqdm(range(pg.psize)):
        p_size = pg.size(i)
        p_nodes = pg.part(i)
        assert (feat_in_order[p_nodes] == feat_shuffled[offset:offset+p_size]).all(), \
            f"Part {i}"
        offset += p_size
    print("Passed")

def scache_from(edge_index, assignments: torch.Tensor, psize: int, serialize=False, data_dir=None):
    pass

class GnnosNodePropPredDataset(BaselineNodePropPredDataset):
    def __init__(self, name, root = 'dataset', partitioner=sampler.MetisMinCutBalanced(), psize=0):
        self.partitioner = partitioner
        self.psize = psize
        self.inmem = False
        super(GnnosNodePropPredDataset, self).__init__(name, root, mmap_feat=False, meta_dict=None)

    # override baseline dataset methods 
    def tensor_from_dict(self, dict, inmem=True, root=None):
        root = self.root if root is None else root
        full_path = osp.join(root, dict['path'])
        shape = dict['shape']
        size = torch.prod(torch.LongTensor(shape)).item()
        dtype = utils.torch_dtype(dict['dtype'])
        # TODO using mmap torch tensor for now
        return torch.from_file(full_path, size=size, dtype=dtype, shared=False).reshape(shape)

    def load_graph(self):
        part_file = osp.join(self.partition_dir, f'p{self.psize}.pt')
        meta_file = osp.join(self.partition_dir, f'p{self.psize}.json')
        if not osp.exists(part_file) or not osp.exists(meta_file):
            # load the original graph to generate partitions
            g = super(GnnosNodePropPredDataset, self).load_graph(self.meta_info['graph'])
            train_idx = self.get_idx_split()['train']
            g.ndata['train_mask'] = torch.zeros(self.num_nodes, dtype=torch.bool)
            g.ndata['train_mask'][train_idx] = True

        # partition assignments
        is_new_partition = False
        if not osp.exists(part_file):
            print(f"Can't find {part_file}. Generate new partitioning? [y/N]")
            ans = input()
            assert ans.lower().startswith('y'), "Exit because no partitioning found"
            os.makedirs(self.partition_dir, exist_ok=True)
            assigns = self.partitioner.partition(g, self.psize)
            torch.save(assigns, part_file)
            is_new_partition = True
        self.assigns = torch.load(part_file)

        # partition graph
        if is_new_partition or not osp.exists(meta_file):
            # graph structures
            data_dir = f'p{self.psize}'
            edge_index = torch.vstack(g.edges())
            with utils.cwd(self.partition_dir):
                os.makedirs(data_dir, exist_ok=True)
                partition_dict, pg = split_graph(edge_index, self.assigns,
                    self.psize, serialize=True, data_dir=f'p{self.psize}')
                del edge_index
            # shuffle labels
            labels = super(GnnosNodePropPredDataset, self).load_labels()
            with utils.cwd(self.partition_dir):
                partition_dict['labels'] = tensor_serialize(labels[pg.src_nids].numpy(),
                    osp.join(data_dir, "labels"))
                del labels
            # shuffle node features
            node_feat = super(GnnosNodePropPredDataset, self).load_node_feat()
            with utils.cwd(self.partition_dir):
                partition_dict['node_feat'] = tensor_serialize(node_feat[pg.src_nids].numpy(),
                    osp.join(data_dir, "node_feat"))
                del node_feat
            del pg
            # serialize metadata
            with utils.cwd(self.partition_dir):
                with open(osp.join(f'p{self.psize}.json'), 'w') as f_meta:
                    f_meta.write(json.dumps(partition_dict, indent=4, cls=utils.DtypeEncoder))
            self.meta_info['partition'] = partition_dict
            del g

        # load partition graph
        with open(meta_file) as f_meta:
            self.partition_dict = json.load(f_meta)
        part_ptr = self.tensor_from_dict(self.partition_dict['part_ptr'],
            self.inmem, root=self.partition_dir)
        src_nids = self.tensor_from_dict(self.partition_dict['src_nids'],
            self.inmem, root=self.partition_dir)
        dst_ptr = self.tensor_from_dict(self.partition_dict['dst_ptr'],
            self.inmem, root=self.partition_dir)
        dst_nids = self.tensor_from_dict(self.partition_dict['dst_nids'],
            self.inmem, root=self.partition_dir)
        return GnnosPartGraph(self.num_nodes, self.psize, part_ptr, src_nids, dst_ptr, dst_nids)

    def load_labels(self):
        return self.tensor_from_dict(self.partition_dict['labels'],
            self.inmem, root=self.partition_dir)
    
    def load_node_feat(self):
        return self.tensor_from_dict(self.partition_dict['node_feat'],
            self.inmem, root=self.partition_dir)

    def load_data(self):
        self.partition_dir = osp.join(self.root, self.partitioner.name)
        self.num_nodes = self.meta_info['num_nodes']
        self.graph = self.load_graph()
        self.labels = self.load_labels()
        self.node_feat = self.load_node_feat()

if __name__ == "__main__":
    dataset_dir = osp.join(os.environ['DATASETS'], 'gnnos')
    name = 'ogbn-products'
    psize = 4096
    data = BaselineNodePropPredDataset(name=name, root=dataset_dir, mmap_feat=False)
    g = data.graph
    node_feat = data.node_feat
    labels = data.labels

    idx = data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    n_train_samples = len(train_nid)
    n_val_samples = len(val_nid)
    n_test_samples = len(test_nid)

    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    in_feats = node_feat.shape[1]

    print(f"""----Data statistics------'
    #Nodes {n_nodes}
    #Edges {n_edges}
    #Classes/Labels (multi binary labels) {n_classes}
    #Train samples {n_train_samples}
    #Val samples {n_val_samples}
    #Test samples {n_test_samples}
    #Labels     {labels.shape}
    #Features   {node_feat.shape}"""
    )

    gnnos_data = GnnosNodePropPredDataset(name=name, root=dataset_dir, psize=psize)
    check_partition(gnnos_data.assigns, gnnos_data.graph)
    check_graph(g, gnnos_data.graph)
    check_feat(node_feat, gnnos_data.graph, gnnos_data.node_feat)
    check_feat(labels, gnnos_data.graph, gnnos_data.labels)

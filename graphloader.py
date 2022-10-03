import os, json
import os.path as osp
from enum import Enum
from functools import namedtuple
import numpy as np
import torch, dgl

import utils, gnnos_utils

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
    def __init__(self, name, root = 'dataset', mmap_feat=False, mmap_graph=False, meta_dict = None):
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
        self.mmap_graph = mmap_graph

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
            edge_index = self.tensor_from_dict(graph_dict['edge_index'], inmem=not self.mmap_graph)
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            utils.using("creating dgl graph (coo)")
            return dgl.graph(data=(src_nodes, dst_nodes),
                num_nodes=self.num_nodes, device='cpu')
        elif graph_dict['format'] in ('csc', 'csr'):
            row_ptr = self.tensor_from_dict(graph_dict['row_ptr'], inmem=not self.mmap_graph)
            col_idx = self.tensor_from_dict(graph_dict['col_idx'], inmem=not self.mmap_graph)
            # if 'edge_ids' in graph_dict:
            #     edge_ids = self.tensor_from_dict(graph_dict['edge_ids'], inmem=not self.mmap_feat)
            # else:
            #     edge_ids = torch.LongTensor()
            edge_ids = torch.LongTensor()
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
import partition_utils
from datasets import tensor_serialize
from gnnos_graph import GnnosPartGraph, GnnosPartGraphCOO, GnnosPartGraphCSF

def coo_to_csf(edge_index):
    '''
    convert an edge index in coo into the compressed fiber (CSF) format
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

def split_graph_coo(num_nodes: int, edge_index: torch.Tensor, edge_assigns: torch.Tensor,
    psize: int):
    '''
    split the given graph into 1D partitions in COO format, based on the edge assignments

    returns GnnosPartGraphCOO, which is a collection of tensors (or gnnos.TensorStore):
    part_ptr: offset array into src_nids & dst_nids for each partitions
    src_nids: source nids
    dst_nids: destination nids
    '''
    # group edges by partition assignments
    sorted_assigns, reverse_map = torch.sort(edge_assigns)
    grouped_index = edge_index[:, reverse_map]
    # get number of edges per partition.
    # NOTE: some partitions could contain 0 edges in rare cases, so we don't want to
    # use unique_consecutive which will skip those partitions
    # _, group_counts = torch.unique_consecutive(sorted_assigns, return_counts=True)
    group_counts = torch.scatter_add(torch.zeros(psize, dtype=torch.long), dim=0,
        index=sorted_assigns, src=torch.ones_like(sorted_assigns))

    part_boundary = torch.empty((psize+1,), dtype=group_counts.dtype)
    part_boundary[0] = 0
    torch.cumsum(group_counts, dim=0, out=part_boundary[1:])
    assert part_boundary[-1].item() == len(edge_index[0])
    return GnnosPartGraph(num_nodes, psize, part_boundary, grouped_index)

def split_graph_csf(num_nodes: int, edge_index: torch.Tensor, edge_assigns: torch.Tensor,
    psize: int):
    '''
    split the given graph into 1D partitions in CSF format, based on the edge assignments

    returns GnnosPartGraphCSF, which is a collection of tensors (or gnnos.TensorStore):
    part_ptr: offset array into src_nids for each partitions
    concatenated CSFs:
        - src_nids: source nids
        - dst_ptr: offset array of dst_nids for each source nid
        - dst_nids: concatenated adjacency lists (sorted) of each source nid
    '''
    split_coo = split_graph_coo(num_nodes, edge_index, edge_assigns, psize)
    part_boundary = split_coo.part_ptr
    grouped_index = split_coo.edge_index

    # accumulate csf from each partition into (part_ptr, src_nids, dst_ptr, dst_nids)
    part_ptr = torch.zeros((psize+1,), dtype=torch.long)
    num_edges = len(edge_index[0])
    src_nids = torch.zeros((num_edges,), dtype=torch.long)
    dst_ptr = torch.zeros((num_edges+1,), dtype=torch.long)
    dst_nids = torch.zeros((len(edge_index[0]),), dtype=torch.long)
    src_nids_idx, dst_ptr_idx, dst_nids_idx = 0, 1, 0
    dst_ptr_off = 0
    for i in tqdm.tqdm(range(psize)):
        p_src, p_dst_ptr, p_dst_nids = coo_to_csf(
            grouped_index[:, part_boundary[i]:part_boundary[i+1]])
        pn, dn = len(p_src), len(p_dst_nids)
        part_ptr[i+1] = src_nids_idx + pn
        src_nids[src_nids_idx : src_nids_idx+pn] = p_src
        dst_ptr[dst_ptr_idx : dst_ptr_idx+pn] = p_dst_ptr + dst_ptr_off
        dst_nids[dst_nids_idx : dst_nids_idx+dn] = p_dst_nids
        src_nids_idx += pn
        dst_ptr_idx += pn
        dst_nids_idx += dn
        if pn != 0:
            dst_ptr_off = p_dst_ptr[-1] + dst_ptr_off
    # use clone to free unused memory
    return GnnosPartGraphCSF(num_nodes, psize, part_ptr, src_nids[:src_nids_idx].clone(),
        dst_ptr[:dst_ptr_idx].clone(), dst_nids[:dst_nids_idx].clone())

# TODO: out-of-core impl
def split_graph_and_serialize(num_nodes: int, edge_index: torch.Tensor, edge_assigns: torch.Tensor,
    psize: int, graph_format='coo', serialize=False, data_dir=None):
    partition_dict = {'graph': {'format': graph_format}}
    graph_dict = partition_dict['graph']
    if graph_format == 'coo':
        pg = split_graph_coo(num_nodes, edge_index, edge_assigns, psize)
        if serialize:
            graph_dict['part_ptr'] = tensor_serialize(
                pg.part_ptr.numpy(), osp.join(data_dir, 'part_ptr'))
            graph_dict['edge_index'] = tensor_serialize(
                pg.edge_index.numpy(), osp.join(data_dir, 'edge_index'))
            return partition_dict, pg 
        else:
            return pg
    elif graph_format == 'csf':
        pg = split_graph_csf(num_nodes, edge_index, edge_assigns, psize)
        if serialize:
            graph_dict['part_ptr'] = tensor_serialize(
                pg.part_ptr.numpy(), osp.join(data_dir, 'part_ptr'))
            graph_dict['src_nids'] = tensor_serialize(
                pg.src_nids.numpy(), osp.join(data_dir, 'src_nids'))
            graph_dict['dst_ptr'] = tensor_serialize(
                pg.dst_ptr.numpy(), osp.join(data_dir, 'dst_ptr'))
            graph_dict['dst_ptr'] = tensor_serialize(
                pg.dst_nids.numpy(), osp.join(data_dir, 'dst_nids'))
            return partition_dict, pg 
        else:
            return pg
    else:
        raise NotImplementedError(f"Unsupported graph format: {graph_format}")

def split_graph_by_src(num_nodes: int, edge_index: torch.Tensor, assignments: torch.Tensor,
    psize: int, graph_format='coo', serialize=False, data_dir=None):
    edge_assigns = assignments.gather(dim=0, index=edge_index[0])
    return split_graph_and_serialize(num_nodes, edge_index, edge_assigns, psize,
        graph_format=graph_format, serialize=serialize, data_dir=data_dir)

def split_graph_by_dst(num_nodes: int, edge_index: torch.Tensor, assignments: torch.Tensor,
    psize: int, graph_format='coo', serialize=False, data_dir=None):
    edge_assigns = assignments.gather(dim=0, index=edge_index[1])
    return split_graph_and_serialize(num_nodes, edge_index, edge_assigns, psize,
        graph_format=graph_format, serialize=serialize, data_dir=data_dir)

def scache_from(g: dgl.DGLGraph, assignments: torch.Tensor, psize: int, k: int,
    serialize=False, data_dir=None):
    '''
    take top-K nodes in terms of degrees and extract the s-cache
    TODO: out-of-core impl
    '''
    degrees = g.in_degrees()
    _, topk_nids = torch.topk(degrees, k)
    new_assignments = assignments.clone()
    new_assignments[topk_nids] = psize # make topk_nids in a new partition
    from_nodes, to_nodes = g.in_edges(topk_nids)
    edge_index = torch.vstack((to_nodes, from_nodes)) # put topk_nids first
    return *split_graph_by_dst(k, edge_index, new_assignments, psize+1,
        serialize=serialize, data_dir=data_dir), topk_nids

class GnnosNodePropPredDataset(BaselineNodePropPredDataset):
    def __init__(self, name, root = 'dataset', partitioner=partition_utils.MetisMinCutBalanced(),
        use_old_feat=False, psize=0, topk=0.01):
        self.partitioner = partitioner
        self.psize = psize
        self.topk = topk
        self.inmem = False
        self.use_old_feat = use_old_feat
        # TODO: out-of-core impl
        super(GnnosNodePropPredDataset, self).__init__(name, root, mmap_feat=False, meta_dict=None)

    # override baseline dataset methods 
    def tensor_from_dict(self, dict, inmem=True, root=None, **kwargs):
        root = self.root if root is None else root
        full_path = osp.abspath(osp.join(root, dict['path']))
        shape = dict['shape']
        size = torch.prod(torch.LongTensor(shape)).item()
        dtype = utils.torch_dtype(dict['dtype'])
        if inmem:
            array = np.fromfile(full_path, dtype=dict['dtype'], count=size)
            return torch.from_numpy(array).reshape(shape)
        else:
            return gnnos_utils.store(full_path, shape, dtype, offset=dict['offset'])
            # return torch.from_file(full_path, size=size, dtype=dtype, shared=True).reshape(shape)

    def load_graph(self):
        part_file = osp.join(self.data_dir, f'p{self.psize}.pt')
        partition_meta = osp.join(self.partition_dir, f'metadata.json')
        scache_meta = osp.join(self.scache_dir, 'metadata.json')
        if not osp.exists(part_file) or not osp.exists(partition_meta) \
            or not osp.exists(scache_meta):
            # load the original graph to generate partitions
            g = super(GnnosNodePropPredDataset, self).load_graph(self.meta_info['graph'])
            train_idx = self.get_idx_split()['train']
            g.ndata['train_mask'] = torch.zeros(self.num_nodes, dtype=torch.bool)
            g.ndata['train_mask'][train_idx] = True
            g_node_feat = super(GnnosNodePropPredDataset, self).load_node_feat()
            g_labels = super(GnnosNodePropPredDataset, self).load_labels()
            # MEM: G + F

        # partition assignments
        is_new_partition = False
        if not osp.exists(part_file):
            print(f"Can't find {part_file}. Generate new partitioning? [y/N]")
            ans = input()
            assert ans.lower().startswith('y'), "Exit because no partitioning file found"
            os.makedirs(self.data_dir, exist_ok=True)
            assigns = self.partitioner.partition(g, self.psize)
            torch.save(assigns, part_file)
            is_new_partition = True
        self.assigns = torch.load(part_file)
        self.parts = partition_utils.partition_from(torch.arange(self.num_nodes),
            self.assigns, self.psize)

        # partition graph
        if is_new_partition or not osp.exists(partition_meta):
            # partitioned graph goes into p_dir
            os.makedirs(self.partition_dir, exist_ok=True)
            with utils.cwd(self.partition_dir):
                edge_index = torch.vstack(g.edges()) # MEM: G * 2 + F
                utils.using("creating partitioned graph")
                partition_dict, _ = split_graph_by_src(self.num_nodes, edge_index,
                    self.assigns, self.psize, serialize=True, data_dir='.') # MEM: G * 3 + F
                del edge_index  # MEM: G * 2 + f
                reordered_nids = torch.cat(self.parts)
                # shuffle labels TODO: out-of-core impl
                partition_dict['labels'] = tensor_serialize(g_labels[reordered_nids].numpy(),
                    'labels')
                # shuffle node features TODO: out-of-core impl
                partition_dict['node_feat'] = tensor_serialize(g_node_feat[reordered_nids].numpy(),
                    'node_feat') # MEM: G * 2 + F * 2
                # serialize metadata
                with open('metadata.json', 'w') as f_meta:
                    f_meta.write(json.dumps(partition_dict, indent=4, cls=utils.DtypeEncoder))

        # generate scache data
        if is_new_partition or not osp.exists(scache_meta):
            # scache goes into scache_dir
            os.makedirs(self.scache_dir, exist_ok=True)
            with utils.cwd(self.scache_dir):
                utils.using("creating scache")
                scache_dict, _, scache_nids = scache_from(g, self.assigns, self.psize,
                    int(self.topk * g.num_nodes()), serialize=True, data_dir='.')
                # MEM: G * 3 + F * 2
                # extract scache node feat
                scache_dict['topk_nids'] = tensor_serialize(scache_nids.numpy(), 'topk_nids')
                scache_dict['labels'] = tensor_serialize(g_labels[scache_nids].numpy(), 'labels')
                scache_dict['node_feat'] = tensor_serialize(
                    g_node_feat[scache_nids].numpy(), 'node_feat')
                # serialize metadata
                with open(osp.join('metadata.json'), 'w') as f_meta:
                    f_meta.write(json.dumps(scache_dict, indent=4, cls=utils.DtypeEncoder))

            del g
            del g_node_feat
            del g_labels

        # load partition graph
        with utils.cwd(self.partition_dir):
            with open('metadata.json') as f_meta:
                self.partition_info = json.load(f_meta)
            graph_info = self.partition_info['graph']
            part_ptr = self.tensor_from_dict(graph_info['part_ptr'], self.inmem, root='.')
            index = gnnos_utils.coo(
                self.tensor_from_dict(graph_info['edge_index'], self.inmem, root='.'),
                self.num_nodes,
            )
            graph = GnnosPartGraph(self.num_nodes, self.psize, part_ptr, index)

        # load scache
        with utils.cwd(self.scache_dir):
            with open('metadata.json') as f_meta:
                self.scache_info = json.load(f_meta)
            scache_graph_info = self.scache_info['graph']
            part_ptr = self.tensor_from_dict(scache_graph_info['part_ptr'], self.inmem, root='.')
            index = gnnos_utils.coo(
                self.tensor_from_dict(scache_graph_info['edge_index'], self.inmem, root='.'),
                self.num_nodes,
            )
            scache_g = GnnosPartGraph(int(self.topk * self.num_nodes), self.psize + 1,
                part_ptr, index)
            # load topk nids and feat into memory
            scache_nids = self.tensor_from_dict(self.scache_info['topk_nids'], True, root='.')
            scache_feat = self.tensor_from_dict(self.scache_info['node_feat'], True, root='.')
            scache_labels = self.tensor_from_dict(self.scache_info['labels'], True, root='.')
        return graph, (scache_g, scache_nids, scache_feat, scache_labels)

    def load_labels(self):
        return self.tensor_from_dict(self.partition_info['labels'],
            self.inmem, root=self.partition_dir)
    
    def load_node_feat(self):
        if self.use_old_feat:
            # use mmap instead for random access
            feat_dict = self.partition_info['node_feat']
            full_path = osp.join(self.partition_dir, feat_dict['path'])
            shape = feat_dict['shape']
            size = torch.prod(torch.LongTensor(shape)).item()
            dtype = utils.torch_dtype(feat_dict['dtype'])
            # shared=False to disable modification of tensors
            tensor = torch.from_file(full_path, size=size, dtype=dtype, shared=False).reshape(shape)
            utils.madvise_random(tensor.data_ptr(), tensor.numel()*tensor.element_size())
            return tensor
        else:
            return self.tensor_from_dict(self.partition_info['node_feat'],
                self.inmem, root=self.partition_dir)

    def load_data(self):
        self.data_dir = osp.join(self.root, self.partitioner.name)
        self.partition_dir = osp.join(self.data_dir, f'p{self.psize}')
        self.scache_dir = osp.join(self.partition_dir, f'scache-{self.topk}')
        self.num_nodes = self.meta_info['num_nodes']
        self.graph, self.scache = self.load_graph()
        self.labels = self.load_labels()
        self.node_feat = self.load_node_feat()

if __name__ == "__main__":
    def check_feat(feat_in_order: torch.Tensor, parts, feat_shuffled: torch.Tensor):
        print("Checking node feat")
        offset = 0
        for i in tqdm.tqdm(range(len(parts))):
            p_size = len(parts[i])
            p_nodes = parts[i]
            assert (feat_in_order[p_nodes] == feat_shuffled[offset:offset+p_size]).all(), \
                f"Partition {i}"
            offset += p_size
        print("Passed")

    def check_labels(labels_ref: torch.Tensor, parts, labels: torch.Tensor):
        print("Checking node labels")
        offset = 0
        for i in tqdm.tqdm(range(len(parts))):
            p_size = len(parts[i])
            p_nodes = parts[i]
            ref = labels_ref[p_nodes]
            ref[torch.isnan(ref)] = -1
            actual = labels[offset:offset+p_size]
            actual[torch.isnan(actual)] = -1
            assert (ref == actual).all(), \
                f"Partition {i}"
            offset += p_size
        print("Passed")

    dataset_dir = osp.join(os.environ['DATASETS'], 'gnnos')
    name = 'ogbn-arxiv'
    psize = 1024

    topk=0.01
    gnnos_data = GnnosNodePropPredDataset(name=name, root=dataset_dir, psize=psize, topk=topk)

    pg = gnnos_data.graph
    assigns = gnnos_data.assigns.clone()
    scache, scache_nids, scache_feat, scache_labels = gnnos_data.scache
    # pg.check_partition(assigns, which='src')
    assigns[scache_nids] = pg.psize
    scache.check_partition(assigns, which='dst')

    data = BaselineNodePropPredDataset(name=name, root=dataset_dir, mmap_feat=False)
    check_feat(data.node_feat, gnnos_data.parts, gnnos_data.node_feat)
    check_labels(data.labels, gnnos_data.parts, gnnos_data.labels)
    check_feat(data.node_feat, [scache_nids], scache_feat)
    check_labels(data.labels, [scache_nids], scache_labels)

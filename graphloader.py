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
            if 'edge_ids' in graph_dict:
                edge_ids = self.tensor_from_dict(graph_dict['edge_ids'], inmem=not self.mmap_feat)
            else:
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
from gnnos_graph import GnnosPartGraph, GnnosPartGraphCOO

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


def split_graph(num_nodes: int, edge_index: torch.Tensor, edge_assigns: torch.Tensor, psize: int):
    '''
    split the given graph into 1D partitions, based on the edge assignments

    returns GnnosPartGraph, which is a collection of tensors (or gnnos.TensorStore):
    part_ptr: offset array into src_nids for each partitions
    src_nids: source nids
    dst_ptr: offset array int dst_nids for each source node
    dst_nids: concatenated adjacency lists (sorted) of each source nid
    '''
    # group edges by partition assignments
    # edge_assigns = assignments.gather(dim=0, index=edge_index[0])
    sorted_assigns, reverse_map = torch.sort(edge_assigns)
    grouped_index = edge_index[:, reverse_map]
    # get number of edges per partition
    # _, group_counts = torch.unique_consecutive(sorted_assigns, return_counts=True)
    group_counts = torch.scatter_add(torch.zeros(psize, dtype=torch.long), dim=0,
        index=sorted_assigns, src=torch.ones_like(sorted_assigns))

    part_boundary = torch.empty((psize+1,), dtype=group_counts.dtype)
    part_boundary[0] = 0
    torch.cumsum(group_counts, dim=0, out=part_boundary[1:])
    assert part_boundary[-1].item() == len(edge_index[0])

    # # accumulate csf from each partition into (part_ptr, src_nids, dst_ptr, dst_nids)
    # part_ptr = torch.zeros((psize+1,), dtype=torch.long)
    # num_edges = len(edge_index[0])
    # src_nids = torch.zeros((num_edges,), dtype=torch.long)
    # dst_ptr = torch.zeros((num_edges+1,), dtype=torch.long)
    # dst_nids = torch.zeros((len(edge_index[0]),), dtype=torch.long)
    # src_nids_idx, dst_ptr_idx, dst_nids_idx = 0, 1, 0
    # dst_ptr_off = 0
    # for i in tqdm.tqdm(range(psize)):
    #     p_src, p_dst_ptr, p_dst_nids = coo_to_csf(grouped_index[:, part_boundary[i]:part_boundary[i+1]])
    #     # print("Partition", i)
    #     # check_graph(num_nodes, edge_index, p_src, [0] + list(p_dst_ptr), p_dst_nids)
    #     pn, dn = len(p_src), len(p_dst_nids)
    #     part_ptr[i+1] = src_nids_idx + pn
    #     src_nids[src_nids_idx : src_nids_idx+pn] = p_src
    #     dst_ptr[dst_ptr_idx : dst_ptr_idx+pn] = p_dst_ptr + dst_ptr_off
    #     dst_nids[dst_nids_idx : dst_nids_idx+dn] = p_dst_nids
    #     src_nids_idx += pn
    #     dst_ptr_idx += pn
    #     dst_nids_idx += dn
    #     if pn != 0:
    #         dst_ptr_off = p_dst_ptr[-1] + dst_ptr_off

    # # use clone to free unused memory
    # return GnnosPartGraph(num_nodes, psize, part_ptr, src_nids[:src_nids_idx].clone(),
    #     dst_ptr[:dst_ptr_idx].clone(), dst_nids[:dst_nids_idx].clone())

    return GnnosPartGraphCOO(num_nodes, psize, part_boundary, grouped_index[0], grouped_index[1])

def split_graph_by_src(num_nodes: int, edge_index: torch.Tensor, assignments: torch.Tensor, psize: int,
    serialize=False, data_dir=None):
    edge_assigns = assignments.gather(dim=0, index=edge_index[0])
    pg = split_graph(num_nodes, edge_index, edge_assigns, psize)
    if serialize:
        partition_dict = {}
        partition_dict['coo_part'] = tensor_serialize(pg.part_ptr.numpy(), osp.join(data_dir, 'coo_part'))
        partition_dict['coo_src'] = tensor_serialize(pg.src_nids.numpy(), osp.join(data_dir, 'coo_src'))
        partition_dict['coo_dst'] = tensor_serialize(pg.dst_nids.numpy(), osp.join(data_dir, 'coo_dst'))
        return partition_dict, pg 
    else:
        return pg

def split_graph_by_dst(num_nodes: int, edge_index: torch.Tensor, assignments: torch.Tensor, psize: int,
    serialize=False, data_dir=None):
    edge_assigns = assignments.gather(dim=0, index=edge_index[1])
    pg = split_graph(num_nodes, edge_index, edge_assigns, psize)
    if serialize:
        partition_dict = {}
        partition_dict['coo_part'] = tensor_serialize(pg.part_ptr.numpy(), osp.join(data_dir, 'coo_part'))
        partition_dict['coo_src'] = tensor_serialize(pg.src_nids.numpy(), osp.join(data_dir, 'coo_src'))
        partition_dict['coo_dst'] = tensor_serialize(pg.dst_nids.numpy(), osp.join(data_dir, 'coo_dst'))
        return partition_dict, pg 
    else:
        return pg

def check_partition(assignments, pg: GnnosPartGraph):
    print("Checking partitions")
    for i in tqdm.tqdm(range(pg.psize)):
        nids = pg.part_src(i)
        assert (assignments[nids] == i).all(), f"Part {i}"
    print("Passed")

def check_graph(g: dgl.DGLGraph, pg: GnnosPartGraph, interval=1):
    print("Checking adjs")
    assert g.num_edges() == len(pg.dst_nids)
    for i, _ in enumerate(tqdm.tqdm(pg.src_nids)):
        if i % interval == 0:
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
        p_nodes = pg.part_src(i)
        assert (feat_in_order[p_nodes] == feat_shuffled[offset:offset+p_size]).all(), \
            f"Part {i}"
        offset += p_size
    print("Passed")

def check_labels(labels_ref: torch.Tensor, pg: GnnosPartGraph, labels: torch.Tensor):
    print("Checking node labels")
    offset = 0
    for i in tqdm.tqdm(range(pg.psize)):
        p_size = pg.size(i)
        p_nodes = pg.part_src(i)
        ref = labels_ref[p_nodes]
        ref[torch.isnan(ref)] = -1
        actual = labels[offset:offset+p_size]
        actual[torch.isnan(actual)] = -1
        assert (ref == actual).all(), f"Part {i}"
        offset += p_size
    print("Passed")


def scache_from(g: dgl.DGLGraph, assignments: torch.Tensor, psize: int, k: int,
    serialize=False, data_dir=None):
    '''
    take top-K nodes in terms of degrees and extract the s-cache
    '''
    degrees = g.in_degrees()
    _, topk_nids = torch.topk(degrees, k)
    new_assignments = assignments.clone()
    new_assignments[topk_nids] = psize # make topk_nids in a new partition
    from_nodes, to_nodes = g.in_edges(topk_nids)
    edge_index = torch.vstack((to_nodes, from_nodes)) # put topk_nids first
    return *split_graph_by_dst(k, edge_index, new_assignments, psize+1,
        serialize=serialize, data_dir=data_dir), topk_nids

def check_scache_partition(assignments, scache: GnnosPartGraph,):
    print("Checking scache partitioning")
    assert scache.psize + 1 == len(scache.part_ptr)
    for i in tqdm.tqdm(range(scache.psize-1)):
        nids = scache.part_dst(i)
        assert (assignments[nids] == i).all(), f"Part {i}"
    print("Passed")

def check_scache(g: dgl.DGLGraph, scache: GnnosPartGraph, k: int):
    print("Checking scache sizes")
    degrees = g.in_degrees()
    topk_degs, topk_nids = torch.topk(degrees, k)
    assert topk_degs.sum().item() == len(scache.dst_nids), \
        f"inconsistent edge number in scache: {len(scache.dst_nids)}"
    sg = g.subgraph(topk_nids)
    sg_edges = scache.part_dst(scache.psize-1)
    assert sg.num_edges() == len(sg_edges), f"{sg.num_edges()} vs {len(sg_edges)}"
    print("Passed")

class GnnosNodePropPredDataset(BaselineNodePropPredDataset):
    def __init__(self, name, root = 'dataset', partitioner=partition_utils.MetisMinCutBalanced(),
        psize=0, topk=0.01):
        self.partitioner = partitioner
        self.psize = psize
        self.topk = topk
        self.inmem = False
        # TODO: make mmap True to save memory
        super(GnnosNodePropPredDataset, self).__init__(name, root, mmap_feat=False, meta_dict=None)

    # override baseline dataset methods 
    def tensor_from_dict(self, dict, inmem=True, root=None, **kwargs):
        root = self.root if root is None else root
        full_path = osp.join(root, dict['path'])
        shape = dict['shape']
        size = torch.prod(torch.LongTensor(shape)).item()
        dtype = utils.torch_dtype(dict['dtype'])
        if inmem:
            array = np.fromfile(full_path, dtype=dict['dtype'], count=size)
            return torch.from_numpy(array).reshape(shape)
        else:
            return gnnos_utils.store(full_path, shape, dtype, offset=dict['offset'])
            # return torch.from_file(full_path, size=size, dtype=dtype, shared=False).reshape(shape)

    def load_graph(self):
        part_file = osp.join(self.partition_dir, f'p{self.psize}.pt')
        meta_file = osp.join(self.partition_dir, f'p{self.psize}.json')
        scache_file = osp.join(self.partition_dir, f'p{self.psize}/scache-{self.topk}.json')
        if not osp.exists(part_file) or not osp.exists(meta_file) or not osp.exists(scache_file):
            # load the original graph to generate partitions
            g = super(GnnosNodePropPredDataset, self).load_graph(self.meta_info['graph'])
            train_idx = self.get_idx_split()['train']
            g.ndata['train_mask'] = torch.zeros(self.num_nodes, dtype=torch.bool)
            g.ndata['train_mask'][train_idx] = True
            g_node_feat = super(GnnosNodePropPredDataset, self).load_node_feat()
            g_labels = super(GnnosNodePropPredDataset, self).load_labels()

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
        self.parts = partition_utils.partition_from(torch.arange(self.num_nodes),
            self.assigns, self.psize)

        # partition graph
        if is_new_partition or not osp.exists(meta_file):
            # graph structures
            data_dir = f'p{self.psize}'
            edge_index = torch.vstack(g.edges())
            with utils.cwd(self.partition_dir):
                os.makedirs(data_dir, exist_ok=True)
                utils.using("creating partitioned graph")
                partition_dict, pg = split_graph_by_src(self.num_nodes, edge_index,
                    self.assigns, self.psize, serialize=True, data_dir=f'p{self.psize}')
                del edge_index

            reordered_nids = torch.cat(self.parts)
            # shuffle labels
            with utils.cwd(self.partition_dir):
                partition_dict['labels'] = tensor_serialize(g_labels[reordered_nids].numpy(),
                    osp.join(data_dir, "labels"))
            # shuffle node features
            with utils.cwd(self.partition_dir):
                partition_dict['node_feat'] = tensor_serialize(g_node_feat[reordered_nids].numpy(),
                    osp.join(data_dir, "node_feat"))
            # serialize metadata
            with open(meta_file, 'w') as f_meta:
                f_meta.write(json.dumps(partition_dict, indent=4, cls=utils.DtypeEncoder))

        # generate scache data
        if is_new_partition or not osp.exists(scache_file):
            scache_dir = f'scache-{self.topk}'
            with utils.cwd(self.partition_dir + f'/p{self.psize}'):
                os.makedirs(scache_dir, exist_ok=True)
                utils.using("creating scache")
                scache_dict, scache, scache_nids = scache_from(g, self.assigns, self.psize,
                    int(self.topk * g.num_nodes()), serialize=True, data_dir=scache_dir)
            # extract scache node feat
            with utils.cwd(self.partition_dir + f'/p{self.psize}'):
                scache_dict['topk_nids'] = tensor_serialize(scache_nids.numpy(),
                    osp.join(scache_dir, "topk_nids"))
                scache_dict['node_feat'] = tensor_serialize(g_node_feat[scache_nids].numpy(),
                    osp.join(scache_dir, "node_feat"))
            # serialize metadata
            with open(osp.join(scache_file), 'w') as f_meta:
                f_meta.write(json.dumps(scache_dict, indent=4, cls=utils.DtypeEncoder))

            del g
            del g_node_feat
            del g_labels

        # load partition graph
        with open(meta_file) as f_meta:
            self.partition_info = json.load(f_meta)
        coo_part = self.tensor_from_dict(self.partition_info['coo_part'],
            self.inmem, root=self.partition_dir)
        coo_src = self.tensor_from_dict(self.partition_info['coo_src'],
            self.inmem, root=self.partition_dir)
        coo_dst = self.tensor_from_dict(self.partition_info['coo_dst'],
            self.inmem, root=self.partition_dir)
        graph = GnnosPartGraphCOO(self.num_nodes, self.psize, coo_part, coo_src, coo_dst)

        # load scache
        with open(scache_file) as f_meta:
            self.scache_info = json.load(f_meta)
        data_dir = osp.join(self.partition_dir, f'p{self.psize}')
        coo_part = self.tensor_from_dict(self.scache_info['coo_part'], self.inmem, root=data_dir)
        coo_src = self.tensor_from_dict(self.scache_info['coo_src'], self.inmem, root=data_dir)
        coo_dst = self.tensor_from_dict(self.scache_info['coo_dst'], self.inmem, root=data_dir)
        scache = GnnosPartGraphCOO(int(self.topk * self.num_nodes), self.psize + 1,
            coo_part, coo_src, coo_dst)
        # load topk nids and feat into memory
        scache_nids = self.tensor_from_dict(self.scache_info['topk_nids'], True, root=data_dir)
        scache_feat = self.tensor_from_dict(self.scache_info['node_feat'], True, root=data_dir)
        return graph, scache, scache_nids, scache_feat


    def load_labels(self):
        return self.tensor_from_dict(self.partition_info['labels'],
            self.inmem, root=self.partition_dir)
    
    def load_node_feat(self):
        return self.tensor_from_dict(self.partition_info['node_feat'],
            self.inmem, root=self.partition_dir)

    def load_data(self):
        self.partition_dir = osp.join(self.root, self.partitioner.name)
        self.num_nodes = self.meta_info['num_nodes']
        self.graph, self.scache, self.scache_nids, self.scache_feat = self.load_graph()
        self.labels = self.load_labels()
        self.node_feat = self.load_node_feat()

if __name__ == "__main__":
    dataset_dir = osp.join(os.environ['DATASETS'], 'gnnos')
    name = 'mag240m'
    psize = 16384

    # data = BaselineNodePropPredDataset(name=name, root=dataset_dir, mmap_feat=False)
    # g = data.graph
    # node_feat = data.node_feat
    # labels = data.labels

    # idx = data.get_idx_split()
    # train_nid = idx['train']
    # val_nid = idx['valid']
    # test_nid = idx['test']
    # n_train_samples = len(train_nid)
    # n_val_samples = len(val_nid)
    # n_test_samples = len(test_nid)

    # n_classes = data.num_classes
    # n_nodes = g.num_nodes()
    # n_edges = g.num_edges()
    # in_feats = node_feat.shape[1]

    # print(f"""----Data statistics------
    # #Nodes {n_nodes}
    # #Edges {n_edges}
    # #Classes/Labels (multi binary labels) {n_classes}
    # #Train samples {n_train_samples}
    # #Val samples {n_val_samples}
    # #Test samples {n_test_samples}
    # #Labels     {labels.shape}
    # #Features   {node_feat.shape}"""
    # )

    topk=0.01
    gnnos_data = GnnosNodePropPredDataset(name=name, root=dataset_dir, psize=psize, topk=topk)
    # check_partition(gnnos_data.assigns, gnnos_data.graph)
    # check_graph(g, gnnos_data.graph, interval=(g.num_nodes()-1)//2000000+1) # check 2e6 nodes
    # check_feat(node_feat, gnnos_data.graph, gnnos_data.node_feat)
    # check_labels(labels, gnnos_data.graph, gnnos_data.labels)
    # check_scache_partition(gnnos_data.assigns, gnnos_data.scache)
    # check_scache(g, gnnos_data.scache, int(topk * gnnos_data.num_nodes))


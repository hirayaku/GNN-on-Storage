import os, json
import os.path as osp
from enum import Enum
from functools import namedtuple
import numpy as np
import torch, dgl

import utils

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

    def tensor_from_dict(self, dict, inmem=True):
        full_path = osp.join(self.root, dict['path'])
        shape = dict['shape']
        size = torch.prod(torch.LongTensor(shape)).item()
        if inmem:
            dtype = utils.torch_dtype(dict['dtype'])
            return torch.from_file(full_path, size=size, dtype=dtype).reshape(shape)
        else:
            return utils.memmap(full_path, random=True, dtype=dict['dtype'], 
                mode='r', offset=0, shape=shape)

    def load_graph(self, graph_dict):
        if graph_dict['format'] == 'coo':
            edge_index = self.tensor_from_dict(graph_dict['edge_index'])
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            utils.using("creating dgl graph")
            return dgl.graph(data=(src_nodes, dst_nodes),
                num_nodes=self.num_nodes, device='cpu')
        elif graph_dict['format'] in ('csc', 'csr'):
            row_ptr = self.tensor_from_dict(graph_dict['row_ptr'])
            col_idx = self.tensor_from_dict(graph_dict['col_idx'])
            if 'edge_ids' in graph_dict:
                edge_ids = self.tensor_from_dict(graph_dict['edge_ids'])
            else:
                edge_ids = torch.LongTensor()
            return dgl.graph(data=('csc', (row_ptr, col_idx, edge_ids)))
        else:
            raise NotImplementedError("Only support coo,csc,csr graph formats")

    def load_labels(self):
        return self.tensor_from_dict(self.meta_info['labels'], inmem=not self.mmap_feat)

    def load_data(self):
        self.num_nodes = self.meta_info['num_nodes']
        self.graph = self.load_graph(self.meta_info['graph'])
        utils.using("graph loaded")
        self.graph = self.graph.formats('csc')
        utils.using("graph transformed")
        self.labels = self.load_labels()
        self.node_feat = self.tensor_from_dict(self.meta_info['node_feat'],
            inmem=not self.mmap_feat)
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


if __name__ == "__main__":
    dataset_dir = osp.join(os.environ['DATASETS'], 'gnnos')
    data = BaselineNodePropPredDataset(name='ogbn-papers100M', root=dataset_dir,
        mmap_feat=True)
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

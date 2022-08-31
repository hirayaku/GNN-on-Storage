#!/usr/bin/python3

import numpy as np
from os import path as osp
import argparse

if __name__ == "__main__":
    '''
    Preprocessor for the ogbn-papers100M dataset:
    after preprocessing, three files are generated
    - .dgl file with graph structure (coo format), masks, and labels
    - .feat file storing node features
    - .shape file storing the shape of node feature numpy array
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ogbn-papers100M', help='Dataset name')
    parser.add_argument('--rootdir', type=str, default='../dataset/', help='Directory to download the OGB dataset')
    parser.add_argument('--graph-output-dir', type=str, help='Directory to store the graph with train/test/val masks')
    parser.add_argument('--feat-output-dir', type=str, help='Directory to store features')
    parser.add_argument('--graph-formats', type=str, default='', help='Graph format (coo, csr or csc)')
    parser.add_argument('--to-bidirected', action='store_true', help='Make the input graph bidirected by adding an in-edge for every out-edge (only make sense when the input is a directed graph)')
    # parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as DGL homogeneous graph.')
    args = parser.parse_args()

    import torch as th
    import dgl
    from dgl.data.utils import save_graphs
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load graph from OGB.')
    data = DglNodePropPredDataset(name=args.dataset, root=args.rootdir)
    keySet = {'train', 'valid', 'test'}
    try:
        splitted_idx = data.get_idx_split()
        assert(set(splitted_idx.keys()) == keySet)
    except:
        print("No train/val/test idx in dataset " + args.dataset)
        raise

    data_dir = osp.join(args.rootdir, data.dir_name)
    if args.graph_output_dir is None:
        args.graph_output_dir = data_dir
    if args.feat_output_dir is None:
        args.feat_output_dir = data_dir

    graph, labels = data[0]

    # node feat -> npy files
    for feat_name in set(graph.ndata.keys()):
        feat_tensor = graph.ndata[feat_name]
        feat_output_path = osp.join(args.feat_output_dir, f'{feat_name}.feat')
        shape_output_path = osp.join(args.feat_output_dir, f'{feat_name}.shape')

        print(f'save feature[{feat_name}] to {feat_output_path}')
        shape_mmap = np.memmap(shape_output_path, mode='w+', dtype='int64',
                               shape=(len(feat_tensor.shape),))
        shape_mmap[:] = np.array(feat_tensor.shape)
        shape_mmap.flush()
        feat_mmap = np.memmap(feat_output_path, mode='w+', dtype='float32',
                              shape=tuple(feat_tensor.shape)) # shape must be tuple
        feat_mmap[:] = feat_tensor[:]
        feat_mmap.flush()

        del graph.ndata[feat_name]

    graph_output_path = osp.join(args.graph_output_dir, 'graph.dgl')
    if args.to_bidirected:
        graph = dgl.to_bidirected(graph)
        #  graph_output_path = osp.join(args.graph_output_dir, 'graph_bidirected.dgl')

    if args.graph_formats != "":
        graph = graph.formats(args.graph_formats.split(','))

    for key in keySet:
        idx = splitted_idx[key]
        # TODO make mask a BoolTensor
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[idx] = True
        graph.ndata[f'{key}_mask'] = mask

    graph.ndata['label'] = labels[:graph.num_nodes()]
    print(f'save graph to {graph_output_path}')
    save_graphs(graph_output_path, [graph])


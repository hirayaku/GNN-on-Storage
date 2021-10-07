#!/usr/bin/python3

import dgl
import numpy as np
import torch as th
from os import path as osp
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.utils import save_graphs

if __name__ == "__main__":
    '''
    Preprocessor for the ogbn-papers100M dataset:
    after preprocessing, two files are generated
    - .dgl file with graph structure (coo format), masks, and labels
    - .npy files storing all node features
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ogbn-papers100M', help='Dataset name')
    parser.add_argument('--rootdir', type=str, default='.', help='Directory to download the OGB dataset')
    parser.add_argument('--graph-output-dir', type=str, help='Directory to store the graph with train/test/val masks')
    parser.add_argument('--graph-formats', type=str, default='csc', help='Graph format (coo, csr or csc)')
    # parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as DGL homogeneous graph.')
    parser.add_argument('--feat-output-dir', type=str, help='Directory to store features')
    args = parser.parse_args()

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
        feat_output_path = osp.join(args.feat_output_dir, f'feat_{feat_name}.npy')

        print(f'save feature[{feat_name}] to {feat_output_path}')
        feat_mmap = np.lib.format.open_memmap(feat_output_path, mode='w+', dtype='float32',
                                              shape=tuple(feat_tensor.shape)) # shape must be tuple!
        feat_mmap[:] = feat_tensor[:]
        feat_mmap.flush()

        del graph.ndata[feat_name]
    
    # TODO: for link prediction, we also need edge feat in npy files
    # TODO: heterogeneous graph with multiple types?

    graph = graph.formats(args.graph_formats.split(','))
    # idx -> mask
    # train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    for key in keySet:
        idx = splitted_idx[key]
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[idx] = True
        graph.ndata[f'{key}_mask'] = mask

    graph.ndata['label'] = labels[:graph.num_nodes()]

    graph_output_path = osp.join(args.graph_output_dir, 'graph.dgl')
    print(f'save graph to {graph_output_path}')
    # TODO: the size of edge data could be comparable to node features
    #       separate nodes with edges
    save_graphs(graph_output_path, [graph])


#!/usr/bin/python3

import numpy as np
from os import path as osp
import argparse
from ogb.lsc import MAG240MDataset
import torch as th
import dgl
from dgl.data.utils import save_graphs
#from ogb.nodeproppred import DglNodePropPredDataset

if __name__ == "__main__":
    '''
    Preprocessor for the mag240m dataset:
    after preprocessing, three files are generated
    - .dgl file with graph structure (coo format), masks, and labels
    - .feat file storing node features
    - .shape file storing the shape of node feature numpy array
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--graph-output-dir', type=str, help='Directory to store the graph with train/test/val masks')
    parser.add_argument('--feat-output-dir', type=str, help='Directory to store features')
    parser.add_argument('--graph-formats', type=str, default='', help='Graph format (coo, csr or csc)')
    parser.add_argument('--to-bidirected', action='store_true', help='Make the input graph bidirected by adding an in-edge for every out-edge (only make sense when the input is a directed graph)')
    #parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as DGL homogeneous graph.')
    args = parser.parse_args()

    print('Building graph MAG240M')
    rootdir = '/jet/home/xhchen/datasets/dgl-data/'
    dataset = MAG240MDataset(root = rootdir)
    print('number of paper nodes: {:d}'.format(dataset.num_papers))
    print('number of author nodes: {:d}'.format(dataset.num_authors))
    print('number of institution nodes: {:d}'.format(dataset.num_institutions))
    print('dimensionality of paper features: {:d}'.format(dataset.num_paper_features))
    print('number of subject area classes: {:d}'.format(dataset.num_classes))

    ei_writes = dataset.edge_index('author', 'writes', 'paper')
    ei_cites = dataset.edge_index('paper', 'paper')
    ei_affiliated = dataset.edge_index('author', 'institution')

    #author_offset = 0
    #inst_offset = author_offset + dataset.num_authors
    #paper_offset = inst_offset + dataset.num_institutions

    graph = dgl.heterograph({('paper', 'cite', 'paper'): (ei_cites[0], ei_cites[1])}, num_nodes_dict={'paper': dataset.num_papers})
    #graph = dgl.heterograph({
            #('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
            #('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
            #('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
            #('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
            #('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
            #})
    print("label type: ", type(dataset.paper_label))
    print("label dtype: ", dataset.paper_label.dtype)
    labels = th.from_numpy(dataset.paper_label)
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    nv = graph.number_of_nodes()
    ne = graph.number_of_edges()
    print('|V|: {}, |E|: {}, #classes: {}'.format(nv, ne, n_classes))
    assert dataset.num_papers == nv

    keySet = {'train', 'valid', 'test-dev', 'test-challenge', 'test-whole'}
    try:
        splitted_idx = dataset.get_idx_split()
        kset = set(splitted_idx.keys())
        print(kset)
        assert(kset == keySet)
    except:
        print("No train/val/test idx")
        raise

    data_dir = osp.join(rootdir, 'mag240m')
    '''
    # features are separated from the graph
    if args.feat_output_dir is None:
        args.feat_output_dir = data_dir
    # node feat -> npy files
    #feat_keys = {'paper-feat'}
    #for feat_name in set(graph.ndata.keys()):
    feat_name = 'paper_feat'
    feat_tensor = dataset.paper_feat
    feat_output_path = osp.join(args.feat_output_dir, f'{feat_name}.feat')
    shape_output_path = osp.join(args.feat_output_dir, f'{feat_name}.shape')

    print(f'save feature[{feat_name}] to {feat_output_path}')
    shape_mmap = np.memmap(shape_output_path, mode='w+', dtype='int64', shape=(len(feat_tensor.shape),))
    shape_mmap[:] = np.array(feat_tensor.shape)
    shape_mmap.flush()
    feat_mmap = np.memmap(feat_output_path, mode='w+', dtype='float32',
                          shape=tuple(feat_tensor.shape)) # shape must be tuple
    feat_mmap[:] = feat_tensor[:]
    feat_mmap.flush()
    #del graph.ndata[feat_name]
    '''
    if args.graph_output_dir is None:
        args.graph_output_dir = data_dir
    if args.to_bidirected:
        graph_output_path = osp.join(args.graph_output_dir, 'graph_bidirected.dgl')
        graph = dgl.to_bidirected(graph)
    else:
        graph_output_path = osp.join(args.graph_output_dir, 'graph.dgl')

    if args.graph_formats != "":
        graph = graph.formats(args.graph_formats.split(','))

    for key in keySet:
        idx = splitted_idx[key]
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[idx] = True
        graph.ndata[f'{key}_mask'] = mask
    graph.ndata['test_mask'] = graph.ndata['test-dev_mask']

    graph.ndata['label'] = labels
    print("graph type: ", type(graph))
    print(graph.ndata.keys())
    print(f'save graph to {graph_output_path}')
    save_graphs(graph_output_path, [graph])


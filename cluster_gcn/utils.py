import os
from functools import namedtuple

import dgl
import numpy as np
import torch
import torch.nn as nn
from dgl.data import PPIDataset
from dgl.data import load_data as _load_data
from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.metrics import f1_score, accuracy_score

class Logger(object):
    '''A custom logger to log stdout to a logging file.'''
    def __init__(self, path):
        """Initialize the logger.

        Paramters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, 'a') as f:
            print(str(s))
            f.write(str(s))

    def writeln(self, s):
        with open(self.path, 'a') as f:
            print(str(s))
            f.write(str(s))
            f.write('\n')

def arg_list(labels):
    hist, indexes, inverse, counts = np.unique(
        labels, return_index=True, return_counts=True, return_inverse=True)
    li = []
    for h in hist:
        li.append(np.argwhere(inverse == h))
    return li

def save_log_dir(args):
    log_dir = './log/{}/{}'.format(args.dataset, args.note)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def calc_f1(y_true, y_pred, multitask):
    if multitask:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")

def calc_acc(y_true, y_pred, multitask):
    if multitask:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)

def evaluate(model, g, feat, labels, mask, multitask=False):
    loss_f = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model.inference(g, feat)
        logits = logits[mask]
        labels = labels[mask]
        f1_mic, f1_mac = calc_f1(labels.cpu().numpy(),
                                 logits.cpu().numpy(), multitask)
        return f1_mic, f1_mac, loss_f(logits, labels)

def load_data(args):
    '''Wraps the dgl's load_data utility to handle ppi special case'''
    DataType = namedtuple('Dataset', ['num_classes', 'g', 'features'])

    if args.feat_mmap:
        dataset_path = os.path.join(args.rootdir, args.dataset.replace('-', '_'))
        graph_path = os.path.join(dataset_path, 'graph.dgl')
        shape_path = os.path.join(dataset_path, 'feat.shape')
        feat_path = os.path.join(dataset_path, 'feat.feat')

        print('load a prepared graph and mmap features')
        (graph,), _ = dgl.load_graphs(graph_path)
        if "valid_mask" in graph.ndata and "val_mask" not in graph.ndata:
            graph.ndata['val_mask'] = graph.ndata['valid_mask']
            del graph.ndata['valid_mask']
        graph.ndata['train_mask'] = graph.ndata['train_mask'] > 0
        graph.ndata['test_mask'] = graph.ndata['test_mask'] > 0
        graph.ndata['val_mask'] = graph.ndata['val_mask'] > 0
        graph.ndata['label'] = graph.ndata['label'].reshape(-1) # to 1-D array
        labels = graph.ndata['label']
        num_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
        # mmap
        shape = np.memmap(shape_path, mode='r', dtype='int64')
        feats = np.memmap(feat_path, mode='r', dtype="float32", shape=tuple(shape))
        data = DataType(g=graph, num_classes=num_classes, features=feats)
        return data

    elif args.dataset.startswith("ogbn"):
        dataset = DglNodePropPredDataset(name=args.dataset, root=args.rootdir)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

        graph.ndata['label'] = labels[:graph.num_nodes()].reshape(-1)
        train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        train_mask[train_idx] = True
        graph.ndata['train_mask'] = train_mask
        val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        val_mask[valid_idx] = True
        graph.ndata['val_mask'] = val_mask
        test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        test_mask[test_idx] = True
        graph.ndata['test_mask'] = test_mask
        n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
        data = DataType(g=graph, num_classes=n_classes, features=graph.ndata['feat'])
        return data

    # datasets used in the original ClusterGCN paper
    if args.dataset != 'ppi':
        dataset = _load_data(args)
        data = DataType(g=dataset[0], num_classes=dataset.num_classes, features=dataset[0].ndata['feat'])
        return data

    train_dataset = PPIDataset('train')
    train_graph = dgl.batch([train_dataset[i] for i in range(len(train_dataset))], edge_attrs=None, node_attrs=None)
    val_dataset = PPIDataset('valid')
    val_graph = dgl.batch([val_dataset[i] for i in range(len(val_dataset))], edge_attrs=None, node_attrs=None)
    test_dataset = PPIDataset('test')
    test_graph = dgl.batch([test_dataset[i] for i in range(len(test_dataset))], edge_attrs=None, node_attrs=None)
    G = dgl.batch(
        [train_graph, val_graph, test_graph], edge_attrs=None, node_attrs=None)

    train_nodes_num = train_graph.number_of_nodes()
    test_nodes_num = test_graph.number_of_nodes()
    val_nodes_num = val_graph.number_of_nodes()
    nodes_num = G.number_of_nodes()
    assert(nodes_num == (train_nodes_num + test_nodes_num + val_nodes_num))
    # construct mask
    mask = np.zeros((nodes_num,), dtype=bool)
    train_mask = mask.copy()
    train_mask[:train_nodes_num] = True
    val_mask = mask.copy()
    val_mask[train_nodes_num:-test_nodes_num] = True
    test_mask = mask.copy()
    test_mask[-test_nodes_num:] = True

    G.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    G.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    G.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=G, num_classes=train_dataset.num_labels, features=G.ndata['feat'])
    return data

def to_torch_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    else:
        return torch.from_numpy(data)

def to_torch_dtype(datatype):
    numpy_to_torch_dtype_dict = {
        np.bool       : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }
    if isinstance(datatype, np.dtype):
        for key in numpy_to_torch_dtype_dict:
            if key == datatype:
                return numpy_to_torch_dtype_dict[key]
    elif isinstance(datatype, torch.dtype):
        return datatype

    raise Exception(f'Datatype is neither numpy or torch dtype: {datatype}')

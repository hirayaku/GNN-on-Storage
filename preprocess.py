import argparse, json, os
import os.path as osp
from collections import namedtuple
from typing import Iterable
import numpy as np

from ogb.nodeproppred import NodePropPredDataset
import utils
from utils import DtypeEncoder

TensorInfo = namedtuple("TensorInfo", ("shape", "dtype", "path", "offset"))

def tensor_serialize(tensor: np.ndarray, path: str):
    byte_tensor = tensor.reshape(-1).view('int8')
    # byte tensor on disk
    disk_tensor = utils.memmap(path, random=False, mode='w+', shape=byte_tensor.shape)
    disk_tensor[:] = byte_tensor[:]
    disk_tensor.flush()
    return TensorInfo(shape=tensor.shape, dtype=str(tensor.dtype),
        path=osp.basename(path), offset=0)._asdict()

def tensor_list_serialize(tensors: Iterable[np.ndarray], path: str) -> Iterable[TensorInfo]:
    byte_sum = sum([t.size * t.itemsize for t in tensors])
    disk_tensor = utils.memmap(path, random=False, mode='w+', shape=(byte_sum,))
    info_list = []
    offset = 0
    for t in tensors:
        bt = t.reshape(-1).view(np.int8)
        disk_tensor[offset:offset+bt.size] = bt[:]
        info_list.append(TensorInfo(shape=t.shape, dtype=str(t.dtype),
            path=osp.basename(path), offset=offset)._asdict())
        offset += bt.size
    disk_tensor.flush()
    assert(offset == byte_sum)
    return info_list

def tensor_deserialize(info: TensorInfo):
    return utils.memmap(info.path, random=False, mode='r',
        dtype=info.dtype, shape=info.shape, offset=info.offset)

_directed_dataset = ['ogbn-arxiv', 'ogbn-papers100M']

def process_ogb(name: str, rootdir: str, add_reverse_edges: bool) -> dict:
    metadata = {"dataset": name}
    metadata["directed"] = name in _directed_dataset

    dataset = NodePropPredDataset(name=name, root=rootdir)
    data_dir = osp.join(rootdir, dataset.dir_name)
    serialize_dir = osp.join(data_dir, "gnnos")
    os.makedirs(serialize_dir, exist_ok=True)

    graph, label = dataset[0]
    if metadata["directed"] and add_reverse_edges:
        edge_index = graph['edge_index']
        edge_index_r = np.array([edge_index[1], edge_index[0]])
        graph['edge_index'] = np.hstack([edge_index, edge_index_r])
        del edge_index, graph_index_r
        metadata["directed"] = False

    for attr, value in graph.items():
        if isinstance(value, np.ndarray):
            attr_file = osp.join(serialize_dir, attr)
            metadata[attr] = tensor_serialize(value, attr_file)
        else:
            metadata[attr] = value
    label_file = osp.join(serialize_dir, 'label')
    metadata['label'] = tensor_serialize(label, label_file)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    idx_file = osp.join(serialize_dir, "idx")
    idx_info = tensor_list_serialize((train_idx, valid_idx, test_idx), idx_file)
    metadata['idx'] = dict(zip(['train', 'valid', 'test'], idx_info))
    metadata['idx']['is_mask'] = False

    # serialize metadata at last
    with open(osp.join(serialize_dir, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, cls=DtypeEncoder))
    return metadata

def process_ogb_lowmem(name: str, rootdir: str, add_reverse_edges: bool) -> dict:
    import gc
    metadata = {"dataset": name}
    metadata["directed"] = name in _directed_dataset

    dataset = NodePropPredDataset(name=name, root=rootdir, save=False)
    data_dir = osp.join(rootdir, dataset.dir_name)
    serialize_dir = osp.join(data_dir, "serialize")
    os.makedirs(serialize_dir, exist_ok=True)

    graph, label = dataset[0]
    edge_index = graph['edge_index']
    del dataset
    del graph
    del label
    gc.collect()

    edge_index_file = osp.join(serialize_dir, 'edge_index')
    if metadata["directed"] and add_reverse_edges:
        edge_index_r = np.array([edge_index[1], edge_index[0]])
        edge_index = np.hstack([edge_index, edge_index_r])
        metadata['edge_index'] = tensor_serialize(edge_index, edge_index_file)
        metadata["directed"] = False
    del edge_index
    gc.collect()

    dataset = NodePropPredDataset(name=name, root=rootdir, save=False)
    graph, label = dataset[0]
    del graph['edge_index']
    gc.collect()

    for attr, value in graph.items():
        if isinstance(value, np.ndarray):
            attr_file = osp.join(serialize_dir, attr)
            metadata[attr] = tensor_serialize(value, attr_file)
        else:
            metadata[attr] = value
    label_file = osp.join(serialize_dir, 'label')
    metadata['label'] = tensor_serialize(label, label_file)

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    idx_file = osp.join(serialize_dir, "idx")
    idx_info = tensor_list_serialize((train_idx, valid_idx, test_idx), idx_file)
    metadata['idx'] = dict(zip(['train', 'valid', 'test'], idx_info))
    metadata['idx']['is_mask'] = False

    # serialize metadata at last
    with open(osp.join(serialize_dir, 'metadata.json'), 'w') as f:
        f.write(json.dumps(metadata, indent=4, cls=DtypeEncoder))
    return metadata

if __name__ == "__main__":
    '''
    Preprocessor for the ogb dataset
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ogbn-papers100M', help='Dataset name')
    parser.add_argument('--rootdir', type=str, default='.',
        help='Directory to download the OGB dataset')
    parser.add_argument('--to-undirected', action='store_true',
        help='Make the input graph (if directed) undirected by adding an in-edge for each out-edge')
    # parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as a homogeneous graph.')
    args = parser.parse_args()

    if args.dataset not in _directed_dataset and args.to_undirected:
        print("not adding reverse edges because the dataset is already undirected")
        args.to_undirected = False

    if args.dataset in ('ogbn-papers100M'):
        process_ogb_lowmem(args.dataset, args.rootdir, args.to_undirected)
    else:
        process_ogb(args.dataset, args.rootdir, args.to_undirected)

import os.path as osp
import json
import torch, gnnos
import numpy as np
from utils import DtypeEncoder

def parse_meta(path):
    with open(osp.join(path, "graph.meta.txt")) as f:
        nv = int(f.readline())
        ne = int(f.readline())
        vbytes, ebytes = [int(tok) for tok in f.readline().split()]
        f.readline()
        feat_sz = int(f.readline())
        nclasses = int(f.readline())
        multilabel = bool(int(f.readline()))
        _, _, train_idx_sz = [int(tok) for tok in f.readline().split()]
        _, _, val_idx_sz = [int(tok) for tok in f.readline().split()]
        _, _, test_idx_sz = [int(tok) for tok in f.readline().split()]
    return nv, ne, vbytes, ebytes, feat_sz, nclasses, multilabel, \
        train_idx_sz, val_idx_sz, test_idx_sz

def new_store(path, shape, dtype, offset=0):
    '''
    return a new TensorStore given the four parameters
    '''
    return gnnos.tensor_store(
        gnnos.options(path).with_shape(shape).with_dtype(dtype).with_offset(offset)
    )

integral_types = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}

def load_oag(path):
    nv, ne, vbytes, ebytes, feat_sz, nclasses, multilabel, \
        train_idx_sz, val_idx_sz, test_idx_sz = parse_meta(path)
    mask_sz = [train_idx_sz, val_idx_sz, test_idx_sz]

    # oag-specific
    feat_dtype = torch.float32
    label_dtype = torch.uint8
    mask_dtype = torch.uint8
    ptr_file = "graph.vertex.bin"
    idx_file = "graph.edge.bin"
    feat_file = "graph.feats.bin"
    label_file = "graph.vlabel.bin"
    mask_files = [f"{name}.masks.bin" for name in ("train", "val", "test")]

    # graph topology
    ptr_store = new_store(osp.join(path, ptr_file), shape=[nv+1], dtype=integral_types[ebytes])
    idx_store = new_store(osp.join(path, idx_file), shape=[ne], dtype=integral_types[vbytes])
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), shape=[nv, feat_sz], dtype=feat_dtype)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), shape=label_shape, dtype=label_dtype)
    # masks
    masks = [new_store(osp.join(path, file), shape=[nv], dtype=mask_dtype).tensor().bool()
        for file in mask_files]

    return graph, feats, multilabel, labels, masks


def read_txt(mask_file) -> torch.Tensor:
    return torch.from_numpy(np.loadtxt(mask_file, dtype='long'))

def idx2mask(num_nodes, idx) -> torch.Tensor:
    '''
    transform index array to mask array
    '''
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask

def load_mag240m(path):
    nv, ne, vbytes, ebytes, feat_sz, nclasses, multilabel, \
        train_idx_sz, val_idx_sz, test_idx_sz = parse_meta(path)
    mask_sz = [train_idx_sz, val_idx_sz, test_idx_sz]

    # mag240M-specific
    feat_dtype = torch.float16
    label_dtype = torch.uint8
    ptr_file = "graph.vertex.bin"
    idx_file = "graph.edge.bin"
    feat_file = "graph.feats.bin"
    label_file = "graph.vlabel.bin"
    mask_files = ['train_idx.txt', 'valid_idx.txt', 'testdev_idx.txt']

    # graph topology
    ptr_store = new_store(osp.join(path, ptr_file), shape=[nv+1], dtype=integral_types[ebytes])
    idx_store = new_store(osp.join(path, idx_file), shape=[ne], dtype=integral_types[vbytes])
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), shape=[nv, feat_sz], dtype=feat_dtype)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), shape=label_shape, dtype=label_dtype)
    # indices
    masks = [read_txt(osp.join(path, mask_f)) for mask_f in mask_files]
    assert [mask.size()[0] for mask in masks] == mask_sz
    masks = [idx2mask(graph.num_nodes, m) for m in masks]

    return graph, feats, multilabel, labels, masks


def load_ogb(path):
    meta_file = osp.join(path, "metadata.json")
    if not osp.exists(meta_file):
        path = osp.join(path, "gnnos")
        meta_file = osp.join(path, "metadata.json")

    def load_by_dict(d):
        tinfo = d.copy()
        # relative path
        tinfo['path'] = osp.join(path, tinfo['path'])
        tinfo['dtype'] = getattr(torch, tinfo['dtype'])
        return new_store(**tinfo)

    with open(meta_file) as f:
        metadata = json.load(f)

    num_nodes = metadata['num_nodes']
    edge_index = load_by_dict(metadata['edge_index'])
    graph = gnnos.COOStore(edge_index.flatten(), num_nodes)
    feats = load_by_dict(metadata['node_feat'])
    multilabel = False
    labels = load_by_dict(metadata['label'])

    is_mask = metadata['idx'].pop('is_mask')
    masks = [ load_by_dict(metadata['idx'][t]).tensor() for t in ('train', 'valid', 'test')]
    if not is_mask:
        masks = [idx2mask(graph.num_nodes, m) for m in masks]
    return graph, feats, multilabel, labels, masks


def metadata2dict(md):
    '''TensorInfo metadata to dict'''
    return {
        "path": md.path, "shape": md.shape,
        "dtype": md.dtype, "offset": md.offset
    }

def create_partitions(graph, feats, labels, partitioning: gnnos.NodePartitions, out_dir):
    '''
    Partition the graph, features and labels based on `partitioning`,
    and save the partition data info as a json file `metadata.json`
    {
        "psize", "assigns": TensorInfo, "index_pos",
        "graph": {"type": "COO", "num_nodes", "src_index", "dst_index"},
        "feats": TensorInfo, "labels": TensorInfo
    }
    '''
    psize = partitioning.psize
    pinfo = {'psize': psize}

    def absf(name):
        return osp.join(out_dir, name)

    # save assignments
    assign_store = gnnos.from_tensor(partitioning.assignments(), absf('assign'))
    pinfo['assigns'] = metadata2dict(assign_store.metadata)

    # save graph
    if isinstance(graph, gnnos.CSRStore):
        pg = gnnos.partition_csr_2d(graph, partitioning)
    else:
        pg = gnnos.partition_coo_2d(graph, partitioning)
    print("Graph partitioned")
    index_pos = gnnos.from_tensor(pg.edge_pos(), absf('index_pos'))
    pinfo['index_pos'] = metadata2dict(index_pos.metadata)
    num_nodes, src_info, dst_info = pg.save(absf('bcoo'))
    pinfo['graph'] = {
        'type': 'COO', 'num_nodes': num_nodes,
        'src_index': metadata2dict(src_info), 'dst_index': metadata2dict(dst_info)
    }

    # save feats and labels
    feat_file = osp.join(out_dir, 'feat')
    shuffled_feats = gnnos.tensor_store(feats.metadata.with_offset(0).with_path(feat_file), "x")
    gnnos.shuffle_store(shuffled_feats, feats, partitioning.nodes())
    print("Feats shuffled")
    pinfo['feats'] = metadata2dict(shuffled_feats.metadata)
    label_file = osp.join(out_dir, 'label')
    shuffled_labels = gnnos.tensor_store(labels.metadata.with_offset(0).with_path(label_file), "x")
    gnnos.shuffle_store(shuffled_labels, labels, partitioning.nodes())
    print("Labels shuffled")
    pinfo['labels'] = metadata2dict(shuffled_labels.metadata)

    with open(absf('metadata.json'), 'w') as fp:
        json.dump(pinfo, fp, indent=4, cls=DtypeEncoder)
    return pg, shuffled_feats, shuffled_labels, partitioning

def load_partitions(path):
    '''
    Load the partitioned graph, features and labels from `input_dir`
    '''
    def load_by_dict(d):
        tinfo = d.copy()
        # relative path
        tinfo['path'] = osp.join(path, tinfo['path'])
        tinfo['dtype'] = getattr(torch, tinfo['dtype'])
        return new_store(**tinfo)

    with open(osp.join(path, 'metadata.json')) as fp:
        pinfo = json.load(fp)

    # load partitioning
    psize = pinfo['psize']
    assignments = load_by_dict(pinfo['assigns']).tensor()
    partitioning = gnnos.node_partitions(psize, assignments)

    # load graph
    index_pos = load_by_dict(pinfo['index_pos']).tensor()
    ginfo = pinfo['graph']
    assert ginfo['type'] == 'COO'
    num_nodes = ginfo['num_nodes']
    src = load_by_dict(ginfo['src_index'])
    dst = load_by_dict(ginfo['dst_index'])
    graph = gnnos.BCOOStore(gnnos.COOStore(src, dst, num_nodes), index_pos, partitioning)

    # load feats and labels
    feats = load_by_dict(pinfo['feats'])
    labels = load_by_dict(pinfo['labels'])

    return graph, feats, labels, partitioning


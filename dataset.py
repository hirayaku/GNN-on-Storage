import os.path as osp
import json
import torch, gnnos

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

def new_store(path, shape, itemsize, offset=0):
    '''
    return a new TensorStore given the four parameters 
    '''
    return gnnos.tensor_store(
        gnnos.options(path).with_shape(shape).with_itemsize(itemsize).with_offset(offset)
    )

def load_oag(path):
    nv, ne, vbytes, ebytes, feat_sz, nclasses, multilabel, \
        train_idx_sz, val_idx_sz, test_idx_sz = parse_meta(path)
    mask_sz = [train_idx_sz, val_idx_sz, test_idx_sz]

    # oag-specific
    feat_bytes = 4
    ptr_file = "graph.vertex.bin"
    idx_file = "graph.edge.bin"
    feat_file = "graph.feats.bin"
    label_file = "graph.vlabel.bin"
    mask_files = [f"{name}.masks.bin" for name in ("train", "val", "test")]

    # graph topology
    ptr_store = new_store(osp.join(path, ptr_file), [nv+1], ebytes)
    idx_store = new_store(osp.join(path, idx_file), [ne], vbytes)
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), [nv, feat_sz], feat_bytes)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), label_shape, 1)
    # masks
    masks = [new_store(osp.join(path, file), [nv], 1).tensor('bool') for file in mask_files]
    
    return graph, feats, multilabel, labels, masks


def read_txt(mask_file) -> torch.Tensor:
    array = []
    with open(mask_file) as f:
        array.append(int(f.readline()))
    return torch.tensor(array, dtype=torch.long)

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
    feat_bytes = 2
    ptr_file = "graph.vertex.bin"
    idx_file = "graph.edge.bin"
    feat_file = "graph.feats.bin"
    label_file = "graph.vlabel.bin"
    mask_files = ['train_idx.txt', 'valid_idx.txt', 'testdev_idx.txt']

    # graph topology
    ptr_store = new_store(osp.join(path, ptr_file), [nv+1], ebytes)
    idx_store = new_store(osp.join(path, idx_file), [ne], vbytes)
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), [nv, feat_sz], feat_bytes)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), label_shape, 1)
    # indices
    masks = [read_txt(osp.join(path, mask_f)) for mask_f in mask_files]
    assert [mask.size()[0] for mask in masks] == mask_sz
    masks = [idx2mask(graph.num_nodes, m) for m in masks]

    return graph, feats, multilabel, labels, masks


def bytes_of(dtype):
    if dtype.endswith('64') or dtype == 'double' or dtype == 'long':
        return 8
    elif dtype.endswith('32') or dtype == 'float' or dtype == 'int':
        return 4
    elif dtype.endswith('16') or dtype == 'short':
        return 2
    elif dtype.endswith('8') or dtype == 'byte' or dtype == 'char':
        return 1
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

def load_ogb(path):
    def load_by_dict(d):
        tinfo = d.copy()
        # relative path
        tinfo['path'] = osp.join(path, tinfo['path'])
        tinfo['itemsize'] = bytes_of(tinfo['dtype'])
        tinfo.pop('dtype')
        return new_store(**tinfo)

    with open(osp.join(path, "metadata.json")) as f:
        metadata = json.load(f)
    
    num_nodes = metadata['num_nodes']
    edge_index = load_by_dict(metadata['edge_index'])
    graph = gnnos.COOStore(edge_index.flatten(), num_nodes)
    feats = load_by_dict(metadata['node_feat'])
    multilabel = False
    labels = load_by_dict(metadata['label'])

    is_mask = metadata['idx'].pop('is_mask')
    dtype = metadata['idx']['train']['dtype']
    masks = [
        load_by_dict(metadata['idx']['train']).tensor(dtype),
        load_by_dict(metadata['idx']['valid']).tensor(dtype),
        load_by_dict(metadata['idx']['test']).tensor(dtype),
    ]
    if not is_mask:
        masks = [idx2mask(graph.num_nodes, m) for m in masks]
    return graph, feats, multilabel, labels, masks

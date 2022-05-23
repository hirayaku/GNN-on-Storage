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

def new_store(path, shape, dtype, offset=0):
    '''
    return a new TensorStore given the four parameters 
    '''
    return gnnos.tensor_store(
        gnnos.options(path).with_shape(shape).with_dtype(dtype).with_offset(offset)
    )

def int_dtype(itemsize: int) -> torch.dtype:
    if itemsize == 1:
        return torch.uint8
    elif itemsize == 2:
        return torch.int16
    elif itemsize == 4:
        return torch.int32
    elif itemsize == 8:
        return torch.int64
    raise ValueError(f"Invalid integral size: {itemsize}")

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
    ptr_store = new_store(osp.join(path, ptr_file), [nv+1], int_dtype(ebytes))
    idx_store = new_store(osp.join(path, idx_file), [ne], int_dtype(vbytes))
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), [nv, feat_sz], feat_dtype)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), label_shape, label_dtype)
    # masks
    masks = [new_store(osp.join(path, file), [nv], mask_dtype).tensor().bool()
        for file in mask_files]
    
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
    feat_dtype = torch.float16
    label_dtype = torch.uint8
    ptr_file = "graph.vertex.bin"
    idx_file = "graph.edge.bin"
    feat_file = "graph.feats.bin"
    label_file = "graph.vlabel.bin"
    mask_files = ['train_idx.txt', 'valid_idx.txt', 'testdev_idx.txt']

    # graph topology
    ptr_store = new_store(osp.join(path, ptr_file), [nv+1], int_dtype(ebytes))
    idx_store = new_store(osp.join(path, idx_file), [ne], int_dtype(vbytes))
    graph = gnnos.CSRStore(ptr_store, idx_store)
    # node features
    feats = new_store(osp.join(path, feat_file), [nv, feat_sz], feat_dtype)
    # labels
    label_shape = [nv, nclasses] if multilabel else [nv, 1]
    labels = new_store(osp.join(path, label_file), label_shape, label_dtype)
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

def dataset_partition(graph, feats, labels, partitioning, out_dir):
    psize = partitioning.psize
    if isinstance(graph, gnnos.CSRStore):
        pg = gnnos.BCOOStore.from_csr_2d(graph, partitioning)
    else:
        pg = gnnos.BCOOStore.from_coo_2d(graph, partitioning)
    print("Partitioning completed")

    feat_file = osp.join(out_dir, f'feat-{psize}')
    label_file = osp.join(out_dir, f'label-{psize}')
    shuffled_feats = gnnos.tensor_store(feats.metadata.with_path(feat_file), "x")
    shuffled_labels = gnnos.tensor_store(labels.metadata.with_path(label_file), "x")
    gnnos.shuffle_store(shuffled_feats, feats, partitioning.nodes())
    gnnos.shuffle_store(shuffled_labels, labels, partitioning.nodes())
    print("Shuffling completed")

    assign_file = osp.join(out_dir, f'assign-{psize}')
    assign_store = gnnos.save(partitioning.assignment(), assign_file)

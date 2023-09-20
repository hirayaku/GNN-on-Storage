import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Reddit
import torch_geometric.transforms as T
from torch_geometric.utils import mask_to_index
from ogb.nodeproppred import PygNodePropPredDataset
from utils import report_mem
from data.io import MmapTensor, TensorMeta, Dtype, DtypeEncoder
from data.io import is_tensor, store_tensor

# load_* returns:
# meta_info: metadata required by graphloader.NodePropPredDataset
# data_dict: data graph & features
# idx: train/val/test idx

def load_reddit(rootdir):
    dataset_dir = osp.join(rootdir, 'reddit')
    dataset = Reddit(root=dataset_dir, pre_transform=T.AddSelfLoops())
    data = dataset[0]
    num_nodes = data.x.shape[0]
    meta_info = {
        'dir_name': 'reddit',
        'num_nodes': num_nodes,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_directed': False,
        'is_hetero': False,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [data.edge_index[0], data.edge_index[1]],
        } ],
        'node_feat': data.x.numpy(),
        'edge_feat': None,
        'num_nodes': num_nodes,
        'labels': data.y.numpy(),
    }
    idx = {
        'train': mask_to_index(data.train_mask).numpy(),
        'valid': mask_to_index(data.val_mask).numpy(),
        'test': mask_to_index(data.test_mask).numpy(),
    }
    return meta_info, data_dict, idx

def load_arxiv(rootdir):
    print("loading arxiv...")
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root=rootdir,
        pre_transform=T.ToUndirected(),
    )
    data = dataset[0]
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': data.num_nodes,
        'num_tasks': dataset.num_tasks,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': dataset.is_hetero,
        'is_directed': False,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [data.edge_index[0], data.edge_index[1]],
        } ] ,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes,
        'labels': data.y,
    }
    idx = dataset.get_idx_split()
    print("arxiv loaded!")
    return meta_info, data_dict, idx

def load_arxiv_r(rootdir):
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root=rootdir,
        pre_transform=T.ToUndirected(),
    )
    data = dataset[0]
    meta_info = {
        'dir_name': 'ogbn_arxiv_r',
        'num_nodes': data.num_nodes,
        'num_tasks': dataset.num_tasks,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': dataset.is_hetero,
        'is_directed': False,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [data.edge_index[0], data.edge_index[1]],
        } ] ,
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes,
        'labels': data.y,
    }
    idx = dataset.get_idx_split()
    rand_idx = torch.randperm(data.num_nodes)
    num_train = int(data.num_nodes * 0.8)
    num_valid = int(data.num_nodes * 0.1)
    idx = {
        'train': rand_idx[:num_train],
        'valid': rand_idx[num_train:num_train+num_valid],
        'test': rand_idx[num_train+num_valid:],
    }
    return meta_info, data_dict, idx

def load_products(rootdir):
    dataset = PygNodePropPredDataset(name='ogbn-products', root=rootdir)
    data = dataset[0]
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': data.num_nodes,
        'num_tasks': dataset.num_tasks,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes, # int(np.max(labels[~np.isnan(labels)])) + 1,
        'is_hetero': dataset.is_hetero,
        'is_directed': False,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index':[data.edge_index[0], data.edge_index[1]], # data.edge_index,
        } ],
        'node_feat': data.x,
        'edge_feat': None,
        'num_nodes': data.num_nodes,
        'labels': data.y,
    }
    idx = dataset.get_idx_split()
    return meta_info, data_dict, idx

def load_papers100m(rootdir):
    # use PygNodeProp... to save memory
    dataset = PygNodePropPredDataset(
        name='ogbn-papers100M', root=rootdir,
        pre_transform=T.ToUndirected(),
    )
    data = dataset[0]
    report_mem("ogbn-papers100M dataset loaded")
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': data.num_nodes,
        'num_tasks': dataset.num_tasks,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': dataset.is_hetero,
        'is_directed': False,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [data.edge_index[0], data.edge_index[1]], # data.edge_index.numpy(),
        } ],
        'node_feat': data.x.numpy(),
        'edge_feat': None,
        'num_nodes': data.num_nodes,
        'labels': data.y.numpy(),
    }
    idx = dataset.get_idx_split()
    return meta_info, data_dict, idx

# load the paper citation subgraph of mag240m dataset
def load_mag240m_c(rootdir):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root=rootdir)
    report_mem("mag240m dataset loaded")
    meta_info = {
        'dir_name': 'mag240m_c',
        'num_nodes': dataset.num_papers,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_directed': False,
        'is_hetero': False,
    }
    src, dst = dataset.edge_index('paper', 'cites', 'paper')
    src, dst = torch.from_numpy(src), torch.from_numpy(dst)
    usrc = torch.cat([src, dst])
    udst = torch.cat([dst, src])
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [usrc, udst], # edge_index,
        } ],
        'node_feat': dataset.paper_feat,
        'edge_feat': None,
        'num_nodes': dataset.num_papers,
        'labels': dataset.paper_label,
    }
    train_idx = dataset.get_idx_split('train')
    valid_idx = dataset.get_idx_split('valid')
    # mag240m doesn't have a proper test set
    idx = {
        'train': train_idx,
        'valid': valid_idx,
        'test': valid_idx,
    }
    return meta_info, data_dict, idx

# load the full mag240m dataset
# full.npy are downloaded from
# https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M
def load_mag240m_f(rootdir):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root=rootdir)

    ei_writes = dataset.edge_index("author", "writes", "paper")
    ei_cites = dataset.edge_index("paper", "paper")
    ei_affiliated = dataset.edge_index("author", "institution")
    author_offset = 0
    inst_offset = author_offset + dataset.num_authors
    paper_offset = inst_offset + dataset.num_institutions
    num_nodes = dataset.num_authors + dataset.num_institutions + dataset.num_papers
    ei_writes[0] += author_offset
    ei_writes[1] += paper_offset
    ei_cites += paper_offset
    ei_affiliated[0] += author_offset
    ei_affiliated[1] += inst_offset
    ei = np.concatenate((ei_writes, ei_cites, ei_affiliated), axis=1)
    ei = np.concatenate(ei[0], ei[1]), np.concatenate(ei[1], ei[0])

    # (graph,), _ = dgl.load_graphs(osp.join(dataset.dir, 'graph.dgl'))
    meta_info = {
        'dir_name': 'mag240m',
        'num_nodes': num_nodes,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        # we made this dataset homogeneous
        'is_hetero': False,
        'is_directed': False,
    }
    feats_info = TensorMeta(
        shape=(num_nodes, dataset.num_paper_features), dtype=Dtype.float16,
        path=osp.join(dataset.dir, 'full.npy')
    ).read_only_().temp_(False)
    feats = MmapTensor(feats_info)
    # feats = np.memmap(osp.join(dataset.dir, 'full.npy'), mode='r', dtype='float16',
    #     shape=(num_nodes, dataset.num_paper_features))
    labels = np.empty(num_nodes, dtype=np.float32)
    labels[:] = np.nan
    labels[paper_offset:] = dataset.paper_label
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [ei[0], ei[1]], # ei,
        } ],
        'node_feat': feats,
        'edge_feat': None,
        'num_nodes': num_nodes,
        'labels': labels,
    }
    train_idx = dataset.get_idx_split('train') + paper_offset
    valid_idx = dataset.get_idx_split('valid') + paper_offset
    idx = {
        'train': train_idx,
        'valid': valid_idx,
        'test': valid_idx,
    }
    report_mem("mag240m dataset loaded")
    return meta_info, data_dict, idx

def load_igb(rootdir):
    from igb.dataloader import IGB260M
    dataset = IGB260M(osp.join(rootdir, "IGB"), size="full", classes=19, in_memory=False, synthetic=False)
    meta_info = {
        'dir_name': 'igb_full',
        'num_nodes': dataset.num_nodes(),
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_directed': False,
        'is_hetero': False,
    }
    edges = dataset.paper_edge
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [edges[:,0], edges[:,1]], # edge_index,
        } ],
        'node_feat': dataset.paper_feat,
        'edge_feat': None,
        'num_nodes': dataset.num_nodes(),
        'labels': dataset.paper_label,
    }
    n_labeled_idx = 227130858
    n_train = int(n_labeled_idx * 0.6)
    n_val   = int(n_labeled_idx * 0.2)
    
    idx = {
        'train': torch.arange(n_train),
        'valid': torch.range(n_train, n_train + n_val),
        'test': torch.arange(n_train + n_val, n_labeled_idx),
    }
    return meta_info, data_dict, idx

def load(name, root):
    load_methods = {
        'reddit': load_reddit,
        'ogbn-arxiv': load_arxiv,
        'ogbn-arxiv-r': load_arxiv_r,
        'ogbn-products': load_products,
        'ogbn-papers100M': load_papers100m,
        'mag240m-c': load_mag240m_c,
        'mag240m-f': load_mag240m_f,
        'mag240m': load_mag240m_f,
        'igb260m': load_igb,
    }
    return load_methods[name](root)

def serialize_data(data: object, dir: str, prefix: str = '', memo={}):
    '''
    serialize an object consists of dictionaries/lists/tuples
    tensor data is written to disk as tensor_store
    data of other types are kept unchanged
    '''
    if isinstance(data, dict):
        metadata = {}
        for k in data:
            metadata[k] = serialize_data(
                data[k], dir, f"{prefix}_{k}" if len(prefix) != 0 else f"{k}", memo
            )
        return metadata
    elif isinstance(data, list) or isinstance(data, tuple):
        return [serialize_data(elem, dir, f"{prefix}_{i}", memo)
            for i, elem in enumerate(data)]
    elif is_tensor(data):
        # using a memo to avoid serialize the same tensor data twice
        data_id = id(data)
        if data_id in memo:
            return memo[data_id]
        else:
            meta = store_tensor(data, osp.join(dir, prefix))
            memo[data_id] = meta
            return meta
    else:
        return data

def serialize(dataset_dict: dict, dir: str) -> dict:
    os.makedirs(dir, exist_ok=True)
    metadata = serialize_data(dataset_dict, dir, memo={})
    with open(osp.join(dir, 'metadata.json'), 'w') as f_meta:
        f_meta.write(
            DtypeEncoder(root=dir, indent=4).encode(metadata)
        )
    return metadata

def transform(name: str, indir: str, outdir: str) -> dict:
    '''
    serialize the dataset into flat binary files
    return a dictionary describing the transformed dataset
    '''
    attr, data, idx = load(name, indir)
    dataset_dict = {
        'attr': attr,
        'data': data,
        'idx': idx,
    }
    serialize_dir = osp.join(outdir, attr['dir_name'])
    return serialize_dir, serialize(dataset_dict, serialize_dir)

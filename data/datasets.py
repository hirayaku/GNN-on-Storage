import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Reddit, Reddit2, Flickr
import torch_geometric.transforms as T
from torch_geometric.utils import mask_to_index
from ogb.nodeproppred import PygNodePropPredDataset
from utils import report_mem
from data.io import MmapTensor, TensorMeta

# load_* returns:
# meta_info: metadata required by graphloader.NodePropPredDataset
# data_dict: data graph & features
# idx: train/val/test idx

def load_flickr(rootdir):
    dataset_dir = osp.join(rootdir, 'flickr')
    dataset = Flickr(root=dataset_dir, pre_transform=T.ToUndirected())
    data = dataset[0]
    num_nodes = data.x.shape[0]
    meta_info = {
        'dir_name': 'flickr',
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
    return meta_info, data_dict, idx

def load_arxiv_r(rootdir):
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', root=rootdir,
        pre_transform=T.ToUndirected(),
    )
    data = dataset[0]
    meta_info = {
        'dir_name': 'ogbn_arxiv_r10',
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
    np.random.seed(1)
    rand_idx = np.random.permutation(data.num_nodes)
    num_train = int(data.num_nodes * 0.1)
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

def load_papers100m_directed(rootdir):
    # use PygNodeProp... to save memory
    dataset = PygNodePropPredDataset(
        name='ogbn-papers100M', root=rootdir,
    )
    data = dataset[0]
    report_mem("ogbn-papers100M (directed) dataset loaded")
    meta_info = {
        'dir_name': dataset.dir_name + "_di",
        'num_nodes': data.num_nodes,
        'num_tasks': dataset.num_tasks,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': dataset.is_hetero,
        'is_directed': True,
    }
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [data.edge_index[0], data.edge_index[1]], # data.edge_index.numpy(),
        } ],
        #  'node_feat': data.x.numpy(),
        'edge_feat': None,
        'num_nodes': data.num_nodes,
        'labels': data.y.numpy(),
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
        # TODO: replace it with mmap tensor of npy file
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
        shape=(num_nodes, dataset.num_paper_features), dtype=torch.float16,
        path=osp.join(dataset.dir, 'full.npy')
    ).temp_(False)
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

def load_igb_large(rootdir):
    from igb.dataloader import IGB260M
    dataset = IGB260M(osp.join(rootdir, "igb"), size="large", classes=19, in_memory=False, synthetic=False)
    meta_info = {
        'dir_name': 'igb_large',
        'num_nodes': dataset.num_nodes(),
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_directed': False,
        'is_hetero': False,
    }
    edges = dataset.paper_edge
    feats_path = osp.join(rootdir, 'igb', 'large', 'processed/paper/node_feat.npy')
    feats_info = TensorMeta(
        shape=(dataset.num_nodes(), 1024), dtype=torch.float32, path=feats_path
    )
    feats_tensor = MmapTensor(feats_info)
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [edges[:,0], edges[:,1]], # edge_index,
        } ],
        'node_feat': feats_tensor,
        'edge_feat': None,
        'num_nodes': dataset.num_nodes(),
        'labels': dataset.paper_label,
    }
    n_labeled_idx = dataset.num_nodes()
    n_train = int(n_labeled_idx * 0.6)
    n_val   = int(n_labeled_idx * 0.2)

    idx = {
        'train': torch.arange(n_train),
        'valid': torch.arange(n_train, n_train + n_val),
        'test': torch.arange(n_train + n_val, n_labeled_idx),
    }
    return meta_info, data_dict, idx

def load_igb260m(rootdir):
    from igb.dataloader import IGB260M
    dataset = IGB260M(osp.join(rootdir, "IGB"), size="full", classes=19, in_memory=False, synthetic=False)
    meta_info = {
        'dir_name': 'igb260m',
        'num_nodes': dataset.num_nodes(),
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_directed': False,
        'is_hetero': False,
    }
    edges = dataset.paper_edge
    feats_path = osp.join(rootdir, 'igb', 'full', 'processed/paper/node_feat.npy')
    feats_info = TensorMeta(
        shape=(dataset.num_nodes(), 1024), dtype=torch.float32, path=feats_path
    )
    feats_tensor = MmapTensor(feats_info)
    data_dict = {
        'graph': [ {
            'format': 'coo',
            'edge_index': [edges[:,0], edges[:,1]], # edge_index,
        } ],
        'node_feat': feats_tensor,
        'edge_feat': None,
        'num_nodes': dataset.num_nodes(),
        'labels': dataset.paper_label,
    }
    n_labeled_idx = 227130858
    n_train = int(n_labeled_idx * 0.6)
    n_val   = int(n_labeled_idx * 0.2)

    idx = {
        'train': torch.arange(n_train),
        'valid': torch.arange(n_train, n_train + n_val),
        'test': torch.arange(n_train + n_val, n_labeled_idx),
    }
    return meta_info, data_dict, idx

def load(name, root):
    load_methods = {
        'ogbn-arxiv': load_arxiv,
        'ogbn-arxiv-r': load_arxiv_r,
        'flickr': load_flickr,
        'reddit': load_reddit,
        'ogbn-products': load_products,
        'ogbn-papers100M-di': load_papers100m_directed,
        'ogbn-papers100M': load_papers100m,
        'mag240m-c': load_mag240m_c,
        'mag240m-f': load_mag240m_f,
        'mag240m': load_mag240m_f,
        'igb-large': load_igb_large,
        'igb260m': load_igb260m,
    }
    return load_methods[name](root)


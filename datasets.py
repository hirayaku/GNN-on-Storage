import os.path as osp
import numpy as np
import torch, dgl
from ogb.nodeproppred import NodePropPredDataset
import utils

# load_* returns:
# meta_info: metadata required by graphloader.BaselineNodePropPredDataset
# data_dict: tensor info (names & location)
# idx: train/val/test idx

def load_reddit(rootdir, self_loop=True):
    dataset = dgl.data.RedditDataset(self_loop=self_loop, raw_dir=rootdir)
    graph = dataset[0]
    meta_info = {
        'dir_name': dataset.name,
        'num_nodes': graph.num_nodes(),
        'add_inverse_edge': False,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': False,
    }
    data_dict = {
        'graph': {
            'format': 'coo',
            'edge_index': [t.numpy() for t in graph.adj_sparse('coo')],
        },
        'node_feat': graph.ndata['feat'].numpy(),
        'edge_feat': None,
        'num_nodes': graph.num_nodes(),
        'labels': graph.ndata['label'].numpy(),
    }
    idx = {
        'train': utils.mask2index(graph.ndata['train_mask']).numpy(),
        'valid': utils.mask2index(graph.ndata['val_mask']).numpy(),
        'test': utils.mask2index(graph.ndata['test_mask']).numpy(),
    }
    return meta_info, data_dict, idx

def load_arxiv(rootdir):
    dataset = NodePropPredDataset(name='ogbn-arxiv', root=rootdir)
    graph, labels = dataset[0]
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': graph['num_nodes'],
        'add_inverse_edge': True,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': False,
    }
    graph['graph'] = {
        'format': 'coo',
        'edge_index': graph['edge_index']
    }
    graph.pop('edge_index')
    graph['labels'] = labels
    idx = dataset.get_idx_split()
    return meta_info, graph, idx

def load_products(rootdir):
    dataset = NodePropPredDataset(name='ogbn-products', root=rootdir)
    graph, labels = dataset[0]
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': graph['num_nodes'],
        'add_inverse_edge': False,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes, # int(np.max(labels[~np.isnan(labels)])) + 1,
        'is_hetero': False,
    }
    graph['graph'] = {
        'format': 'coo',
        'edge_index': graph['edge_index']
    }
    graph.pop('edge_index')
    graph['labels'] = labels
    idx = dataset.get_idx_split()
    return meta_info, graph, idx

def load_papers100m(rootdir):
    dataset = NodePropPredDataset(name='ogbn-papers100M', root=rootdir)
    graph, labels = dataset[0]
    meta_info = {
        'dir_name': dataset.dir_name,
        'num_nodes': graph['num_nodes'],
        'add_inverse_edge': True,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': False,
    }
    graph['graph'] = {
        'format': 'coo',
        'edge_index': graph['edge_index']
    }
    graph.pop('edge_index')
    graph['labels'] = labels
    idx = dataset.get_idx_split()
    utils.using("ogbn-papers100M dataset loaded")
    return meta_info, graph, idx

# load the paper citation subgraph of mag240m dataset
def load_mag240m_c(rootdir):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root=rootdir)
    meta_info = {
        'dir_name': 'mag240m_c',
        'num_nodes': dataset.num_papers,
        'add_inverse_edge': True,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': False,
    }
    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    data_dict = {
        'graph': {
            'format': 'coo',
            'edge_index': edge_index,
        },
        'node_feat': dataset.paper_feat,
        'edge_feat': None,
        'num_nodes': dataset.num_papers,
        'labels': dataset.paper_label,
    }
    idx = dataset.get_idx_split()
    # mag240m doesn't have a proper test set
    idx['test'] = idx['valid']
    return meta_info, data_dict, idx

# load the full mag240m dataset
# graph.dgl & full.npy are downloaded from
# https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/MAG240M
def load_mag240m_f(rootdir):
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root=rootdir)
    (graph,), _ = dgl.load_graphs(osp.join(dataset.dir, 'graph.dgl'))
    meta_info = {
        'dir_name': 'mag240m',
        'num_nodes': graph.num_nodes(),
        'add_inverse_edge': False,
        'num_tasks': 1,
        'task_type': 'multiclass classification',
        'num_classes': dataset.num_classes,
        'is_hetero': True,
    }
    feats = np.memmap(osp.join(dataset.dir, 'full.npy'), mode='r', dtype='float16',
        shape=(graph.num_nodes(), dataset.num_paper_features))
    row_ptr, col_idx, eids = graph.adj_sparse('csc')
    data_dict = {
        'graph': {
            'format': 'csc',
            'row_ptr': row_ptr.numpy(),
            'col_idx': col_idx.numpy(),
            'edge_ids': eids.numpy() if eids is not None else None,
        },
        'node_feat': feats,
        'edge_feat': None,
        'edge_type': graph.edata['etype'].numpy(),
        'num_nodes': graph.num_nodes(),
        'labels': dataset.paper_label,
    }
    paper_offset = dataset.num_authors + dataset.num_institutions
    train_idx = dataset.get_idx_split('train') + paper_offset
    valid_idx = dataset.get_idx_split('valid') + paper_offset
    idx = {
        'train': train_idx,
        'valid': valid_idx,
        'test': valid_idx,
    }
    utils.using("mag240m dataset loaded")
    return meta_info, data_dict, idx

def load(name, root):
    load_methods = {
        'reddit': load_reddit,
        'ogbn-arxiv': load_arxiv,
        'ogbn-products': load_products,
        'ogbn-papers100M': load_papers100m,
        'mag240m-c': load_mag240m_c,
        'mag240m-f': load_mag240m_f,
        'mag240m': load_mag240m_f,
    }
    return load_methods[name](root)


from collections import namedtuple
import os, json

# serialize the loaded dataset into custom formats

TensorInfo = namedtuple("TensorInfo", ("shape", "dtype", "path", "offset"))

def tensor_serialize(tensor: np.ndarray, path: str):
    tensor.tofile(path)
    return TensorInfo(shape=tensor.shape, dtype=str(tensor.dtype),
        path=osp.relpath(path), offset=0)._asdict()

def serialize_data(name: str, rootdir: str, outdir: str) -> dict:
    meta_info, data_dict, idx = load(name, rootdir)

    serialize_dir = osp.join(outdir, meta_info['dir_name'])
    os.makedirs(serialize_dir, exist_ok=True)

    with utils.cwd(serialize_dir):
        # serialize graph
        graph_dict = data_dict['graph']
        meta_info['graph'] = {'format': graph_dict['format']}
        if graph_dict['format'] == 'coo':
            path = 'edge_index'
            src, dst = graph_dict['edge_index']
            num_edges = len(src)
            if meta_info['add_inverse_edge']:
                assert meta_info['is_hetero'] is False, "Can't add inverse edges to heterographs"
                assert data_dict['edge_feat'] is None, "Can't add inverse edges to graph with edge features"
                disk_index = utils.memmap(path, dtype=src.dtype,
                    mode='w+', shape=(4, num_edges))
                disk_index[:num_edges] = src[:]
                disk_index[num_edges:2*num_edges] = dst[:]
                disk_index[2*num_edges:3*num_edges] = dst[:]
                disk_index[3*num_edges:] = src[:]
                disk_index.flush()
                meta_info['graph']['edge_index'] = TensorInfo(shape=(2, 2*num_edges),
                    dtype=str(src.dtype), path=path, offset=0)._asdict()
            else:
                disk_index = utils.memmap(path, dtype=src.dtype,
                    mode='w+', shape=(2, num_edges))
                disk_index[:num_edges] = src[:]
                disk_index[num_edges:] = dst[:]
                disk_index.flush()
                meta_info['graph']['edge_index'] = TensorInfo(shape=(2, num_edges),
                    dtype=str(src.dtype), path=path, offset=0)._asdict()
            # other tensors if any
            for attr, value in graph_dict.items():
                if attr != 'format' and attr != 'edge_index':
                    meta_info['graph'][attr] = tensor_serialize(value, attr)

        elif graph_dict['format'] in ('csc', 'csr'):
            # csc, csr
            for attr, value in graph_dict.items():
                if attr != 'format':
                    meta_info['graph'][attr] = tensor_serialize(value, attr)
        
        else:
            raise NotImplementedError("Only support coo,csc,csr graph formats")
            
        meta_info.pop('add_inverse_edge')

        # serialize other tensors in data_dict: node_feat, edge_feat, edge_type, labels
        for attr, value in data_dict.items():
            if attr == 'graph':
                continue
            if isinstance(value, np.ndarray):
                meta_info[attr] = tensor_serialize(value, attr)
            else:
                meta_info[attr] = value
        
        # serialize idx
        meta_info['idx'] = {}
        for attr, idx_tensor in idx.items():
            meta_info['idx'][attr] = tensor_serialize(idx_tensor, f'{attr}-idx')
        
        # serialize metadata
        with open('metadata.json', 'w') as f_meta:
            f_meta.write(json.dumps(meta_info, indent=4, cls=utils.DtypeEncoder))
        
        return meta_info

if __name__ == "__main__":
    import os
    print(load('reddit', os.environ['DATASETS']))

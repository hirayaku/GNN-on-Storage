import dgl
import torch as th
import os.path as osp
from ogb.lsc import MAG240MDataset

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_labels

def load_ogb(name, root='dataset'):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name, root=root)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]

    #  ndata = dict()
    #  for k, v in graph.ndata.items():
    #      ndata[k] = v
    #  if 'products' not in name:
    #      graph = dgl.to_bidirected(graph)
    #  for k, v in ndata.items():
    #      graph.ndata[k] = v
    #  dgl.save_graphs(osp.join(osp.join(root, name), "processed/dgl_data_bidrected"), [graph], {'labels': labels})

    in_feats = graph.ndata['feat'].shape[1]
    labels = labels[:, 0]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    labels = labels.long()
    graph.ndata['label'] = labels

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    print('finish constructing', name)
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g

def load_mag240m(rootdir='/jet/home/xhchen/datasets/dgl-data/'):
    print("Loading graph MAG240M from", rootdir)
    dataset = MAG240MDataset(root=rootdir)
    print('number of paper nodes: {:d}'.format(dataset.num_papers))
    print('dimensionality of paper features: {:d}'.format(dataset.num_paper_features))
    print('number of subject area classes: {:d}'.format(dataset.num_classes))
    ei_cites = dataset.edge_index('paper', 'paper')
    graph = dgl.graph((ei_cites[0], ei_cites[1]), num_nodes=dataset.num_papers)
    labels = th.from_numpy(dataset.paper_label)
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))-1 # exclude nan and -1
    labels = labels.long()
    nv = graph.number_of_nodes()
    ne = graph.number_of_edges()
    print('|V|: {}, |E|: {}, #classes: {}'.format(nv, ne, n_classes))
    graph.ndata['label'] = labels
    keySet = {'train', 'valid', 'test-dev', 'test-challenge', 'test-whole'}
    try:
        splitted_idx = dataset.get_idx_split()
        kset = set(splitted_idx.keys())
        print(kset)
        assert(kset == keySet)
    except:
        print("No train/val/test idx")
        raise
    for key in keySet:
        idx = splitted_idx[key]
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[idx] = True
        graph.ndata[f'{key}_mask'] = mask

    graph.ndata['test_mask'] = graph.ndata['test-dev_mask']
    feats = dataset.paper_feat
    print("type of feats is : ", type(feats))
    print("dtype of feats is : ", feats.dtype)
    feat_tensor = th.tensor(feats)
    print("type of feat_tensor is : ", type(feat_tensor))
    print("dtype of feat_tensor is : ", feat_tensor.dtype)
    #feat_tensor = feat_tensor.float()
    #print("after casting, dtype of feat_tensor is : ", feat_tensor.dtype)
    graph.ndata['feat'] = feat_tensor
    print("type of labels is : ", type(labels))
    print("dtype of labels is : ", labels.dtype)
    return graph, n_classes

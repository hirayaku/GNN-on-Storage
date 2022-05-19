import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
#from ogb.nodeproppred import DglNodePropPredDataset
import argparse
import tqdm
from sklearn.metrics import f1_score
from pyinstrument import Profiler
import os, time
import os.path as osp

import sys
sys.path.append(os.path.abspath("../../"))
import utils

sys.path.append(os.path.abspath("../graphsage/"))
from load_graph import load_reddit, load_ogb, inductive_split

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def run(args, device, data):
    #dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    #graph = dataset[0]      # already prepares ndata['label'/'train_mask'/'val_mask'/'test_mask']
    #feats = graph.ndata['feat']
    #labels = graph.ndata['label']
    #num_classes = dataset.num_classes

    num_classes, graph, feats, labels = data
    in_feats = feats.shape[1]
 
    model = SAGE(in_feats, args.num_hidden, num_classes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    print("setup sampler")
    num_partitions = 1000
    if args.disk_feat:
        sampler = dgl.dataloading.ClusterGCNSampler(graph, num_partitions,
                  prefetch_ndata=['label', 'train_mask', 'valid_mask', 'test_mask'])
    #          prefetch_ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
    else:
        sampler = dgl.dataloading.ClusterGCNSampler(graph, num_partitions,
                  prefetch_ndata=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])

    print("setup data loader")
    print("device: ", device)
    # DataLoader for generic dataloading with a graph, a set of indices (any indices, like
    # partition IDs here), and a graph sampler.
    # NodeDataLoader and EdgeDataLoader are simply special cases of DataLoader where the
    # indices are guaranteed to be node and edge IDs.
    #dataloader = dgl.dataloading.DataLoader(
    dataloader = dgl.dataloading.NodeDataLoader(
        graph, torch.arange(num_partitions).to(device), 
        sampler, device=device, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0)
        #shuffle=True, drop_last=False, num_workers=0, use_uva=True)

    print("start training")
    durations = []
    iter_tput = []
    for epoch in range(args.num_epochs):
        profiler = Profiler()
        profiler.start()
        t0 = time.time()
        model.train()
        tic_step = time.time()
        for it, sg in enumerate(dataloader):
            x = sg.ndata['feat']
            y = sg.ndata['label']
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            m = sg.ndata['train_mask'].bool()
            y_hat = model(sg, x)
            loss = F.cross_entropy(y_hat[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            iter_tput.append(args.batch_size / (time.time() - tic_step))
            if it % args.log_every == 0:
                acc = MF.accuracy(y_hat[m], y[m])
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                avg_tput = np.mean(iter_tput[3:]) if (it > 2 or epoch > 0) else iter_tput[it]
                #print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, it, loss.item(), acc.item(), avg_tput, gpu_mem_alloc))
            tic_step = time.time()

        tt = time.time()
        epoch_time = tt - t0
        print('Epoch Time(s): {:.4f}'.format(epoch_time))
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        durations.append(epoch_time)

        model.eval()
        with torch.no_grad():
            val_preds, test_preds = [], []
            val_labels, test_labels = [], []
            for it, sg in enumerate(dataloader):
                x = sg.ndata['feat']
                y = sg.ndata['label']
                m_val = sg.ndata['val_mask'].bool()
                m_test = sg.ndata['test_mask'].bool()
                y_hat = model(sg, x)
                val_preds.append(y_hat[m_val])
                val_labels.append(y[m_val])
                test_preds.append(y_hat[m_test])
                test_labels.append(y[m_test])
            val_preds = torch.cat(val_preds, 0)
            val_labels = torch.cat(val_labels, 0)
            test_preds = torch.cat(test_preds, 0)
            test_labels = torch.cat(test_labels, 0)
            val_acc = MF.accuracy(val_preds, val_labels)
            test_acc = MF.accuracy(test_preds, test_labels)
            print('Validation acc:', val_acc.item(), 'Test acc:', test_acc.item())
            print('Eval Acc {:.4f}'.format(val_acc))
            print('Test Acc: {:.4f}'.format(test_acc))

    print(np.mean(durations[4:]), np.std(durations[4:]))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--rootdir', type=str, default='../dataset/')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=100)
    argparser.add_argument('--log-every', type=int, default=2)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--disk-feat', action='store_true', help="Put features on disk")
    args = argparser.parse_args()

    print(f'DGL version {dgl.__version__} from {dgl.__path__}')

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    if args.disk_feat:
        dataset_dir = args.rootdir + args.dataset
        print(dataset_dir)
        graphs, _ = dgl.load_graphs(osp.join(dataset_dir, "graph.dgl"))
        g = graphs[0]
        for k, v in g.ndata.items():
            if k.endswith('_mask'):
                g.ndata[k] = v.bool()
        feat_shape_file = osp.join(dataset_dir, "feat.shape")
        feat_file = osp.join(dataset_dir, "feat.feat")
        shape = tuple(utils.memmap(feat_shape_file, mode='r', dtype='int64', shape=(2,)))
        feats = utils.memmap(feat_file, random=True, mode='r', dtype='float32', shape=shape)
        feat_len = feats.shape[1]
        labels = g.ndata['label']
        n_classes = torch.max(g.ndata['label']).item() + 1
    else:
        if args.dataset == 'reddit':
            g, n_classes = load_reddit()
        elif args.dataset.startswith('ogbn'):
            g, n_classes = load_ogb(name=args.dataset, root=args.rootdir)
        else:
            raise Exception('unknown dataset')
        feat_len = g.ndata['features'].shape[1]
        #feats = g.ndata.pop('features')
        feats = g.ndata['features']
        #print("The type of features is : ", type(g.ndata['features']))
        #labels = g.ndata.pop('labels')
        labels = g.ndata['labels']
        g.ndata['label'] = g.ndata['labels']
        g.ndata['feat'] = g.ndata['features']

    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    print('|V|: {}, |E|: {}, #classes: {}, feat_length: {}'.format(nv, ne, n_classes, feat_len))

    #if not args.data_cpu:
    #    feats = feats.to(device)
    #    labels = labels.to(device)

    g.create_formats_()
    data = n_classes, g, feats, labels
    run(args, device, data)

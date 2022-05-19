import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
import time
import numpy as np
import argparse
import tqdm
from sklearn.metrics import f1_score
from pyinstrument import Profiler
import os, time
import os.path as osp

import os, sys
sys.path.append(os.path.abspath("../../"))
import utils

sys.path.append(os.path.abspath("../graphsage/"))
from load_graph import load_reddit, load_ogb, inductive_split
from graphsage import *

def run(args, device, data):
    num_classes, graph, feats, labels = data
    in_feats = feats.shape[1]
    #train_nid = th.nonzero(graph.ndata['train_mask'], as_tuple=True)[0]
    #if args.disk_feat:
    #    val_nid = th.nonzero(graph.ndata['valid_mask'], as_tuple=True)[0]
    #    test_nid = th.nonzero(~(graph.ndata['train_mask'] | graph.ndata['valid_mask']), as_tuple=True)[0]
    #else:
    #    val_nid = th.nonzero(graph.ndata['val_mask'], as_tuple=True)[0]
    #    test_nid = th.nonzero(~(graph.ndata['train_mask'] | graph.ndata['val_mask']), as_tuple=True)[0]

    model = SAGE1(in_feats, args.num_hidden, num_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    opt = th.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

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
    uva = True
    if device == th.device('cpu'):
        uva = False
    dataloader = dgl.dataloading.NodeDataLoader(
    #dataloader = dgl.dataloading.DataLoader(
        graph, th.arange(num_partitions).to(device), 
        sampler, device=device, batch_size=args.batch_size,
        shuffle=True, drop_last=False, num_workers=0, use_uva=uva)

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
            batch_labels = sg.ndata['label']
            m = sg.ndata['train_mask'].bool()
            if args.disk_feat:
                input_nodes = sg.ndata[dgl.NID]
                if device == th.device('cuda'):
                    input_nodes.cpu()
                batch_inputs = feats[input_nodes].to(device)
                batch_labels = batch_labels.reshape(-1,)
                #batch_labels = labels[input_nodes]
            else:
                batch_inputs = sg.ndata['feat']
            batch_pred = model(sg, batch_inputs)
            loss = F.cross_entropy(batch_pred[m], batch_labels[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            iter_tput.append(len(batch_labels[m]) / (time.time() - tic_step))
            if it % args.log_every == 0:
                acc = MF.accuracy(batch_pred[m], batch_labels[m])
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                avg_tput = np.mean(iter_tput[3:]) if (it > 2 or epoch > 0) else iter_tput[it]
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
        with th.no_grad():
            val_preds, test_preds = [], []
            val_labels, test_labels = [], []
            for it, sg in enumerate(dataloader):
                if args.disk_feat:
                    x = feats[sg.ndata[dgl.NID]].to(device)
                    m_val = sg.ndata['valid_mask'].bool()
                else:
                    x = sg.ndata['feat']
                    m_val = sg.ndata['val_mask'].bool()
                y = sg.ndata['label']
                m_test = sg.ndata['test_mask'].bool()
                y_hat = model(sg, x)
                val_preds.append(y_hat[m_val])
                val_labels.append(y[m_val])
                test_preds.append(y_hat[m_test])
                test_labels.append(y[m_test])
            val_preds = th.cat(val_preds, 0)
            val_labels = th.cat(val_labels, 0)
            test_preds = th.cat(test_preds, 0)
            test_labels = th.cat(test_labels, 0)
            val_acc = MF.accuracy(val_preds, val_labels)
            test_acc = MF.accuracy(test_preds, test_labels)
            #print('Validation acc:', val_acc.item(), 'Test acc:', test_acc.item())
            print('Validation Acc {:.4f}'.format(val_acc))
            print('Test Acc: {:.4f}'.format(test_acc))

    print('Average epoch time: {:.4f}'.format(np.mean(durations[1:])))
    #print('Average : {:.4f}'.format(np.std(durations[1:])))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--rootdir', type=str, default='../dataset/')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=3)
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
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    if args.disk_feat:
        print("reading features from disk")
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
        n_classes = th.max(g.ndata['label']).item() + 1
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
        #labels = g.ndata.pop('labels')
        labels = g.ndata['labels']
        g.ndata['label'] = g.ndata['labels']
        g.ndata['feat'] = g.ndata['features']

    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    print('|V|: {}, |E|: {}, #classes: {}, feat_length: {}'.format(nv, ne, n_classes, feat_len))

    if not args.data_cpu:
        feats = feats.to(device)
        labels = labels.to(device)

    g.create_formats_()
    data = n_classes, g, feats, labels
    run(args, device, data)

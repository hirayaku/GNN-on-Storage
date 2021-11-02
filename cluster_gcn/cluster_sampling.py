import argparse
import os
import time
import random
from pyinstrument import Profiler

import numpy as np
import networkx as nx
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data.utils import get_download_dir

from modules import GraphSAGE
from sampler import ClusterIter
from utils import Logger, evaluate, calc_f1, save_log_dir, load_data, to_torch_tensor

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    data = load_data(args)
    g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    feats = data.features

    # train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]

    # Normalize features
    if args.normalize:
        assert args.feat_mmap is False, "can't normalize mmapped features"
        train_feats = feats[train_mask]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats.data.numpy())
        # FIXME: g.ndata['feat'] not set to feats
        feats = to_torch_tensor(scaler.transform(feats.data.numpy()))

    in_feats = feats.shape[1]
    n_classes = data.num_classes
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------
    #Edges %d
    #Classes %d
    #Train samples %d (%.2f%%)
    #Val samples %d (%.2f%%)
    #Test samples %d (%.2f%%)""" %
            (n_edges, n_classes,
            n_train_samples, n_train_samples / n_nodes * 100,
            n_val_samples, n_val_samples / n_nodes * 100,
            n_test_samples, n_test_samples / n_nodes * 100))
    print("    labels shape   ", g.ndata['label'].shape)
    print("    features shape ", feats.shape)

    # create GCN model
    if args.self_loop and not args.dataset.startswith('reddit'):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        print("adding self-loop edges")
    # metis only support int64 graph
    g = g.long()

    # FIXME:
    # nodes in `train_nid` is used to induce a subgraph from the full graph `g`
    # it works well when training nodes consist of a large portion of all nodes
    # however, when len(train_nid) << g.num_nodes(), a significant amount of edges
    # are dropped - i.e. missing information
    cluster_iterator = ClusterIter(
        args.dataset, g, args.psize, args.batch_clusters, train_nid, use_pp=args.use_pp)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.int().to(args.gpu)

    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.use_pp,
                      full_batch=False)

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.writeln(args)

    # Loss function
    if multitask:
        print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss()
    else:
        print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1
    device = train_nid.device

    for epoch in range(args.n_epochs):
        if args.profile:
            profiler = Profiler()
            profiler.start()
        for j, cluster in enumerate(cluster_iterator):
            # sync with upper level training graph
            if cuda:
                cluster = cluster.to(torch.cuda.current_device())
            cluster_feats = to_torch_tensor(feats[cluster.nodes()]) if args.feat_mmap else cluster.ndata['feat']
            cluster_labels = cluster.ndata['label']

            # GraphSAGE sampler
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in args.fan_out.split(',')][:args.n_layers])
            dataloader = dgl.dataloading.NodeDataLoader(
                cluster,
                cluster.nodes(),
                sampler,
                batch_size=args.batch_nodes,
                shuffle=True,
                drop_last=False,
                num_workers=0) # TODO: nonzero num_works brings huge slowdown, why?

            steps = len(dataloader)
            log_iter =  j != 0 and (j % args.log_every == 0 or j+1==len(cluster_iterator))

            model.train()
            for step, (input_nodes, train_nodes, blocks) in enumerate(dataloader):
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(cluster_feats, cluster_labels,
                                                            train_nodes, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_f(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if log_iter:
                    logger.writeln(f"#input_nodes {input_nodes.shape[0]}, #train_nodes {train_nodes.shape[0]}")
                    f1_mic, f1_mac = calc_f1(batch_labels.detach().numpy(),
                                            batch_pred.detach().numpy(), multitask=False)
                    iter_msg = (f"epoch:{epoch}/{args.n_epochs}, "
                                f"iteration:{j+1}/{len(cluster_iterator)}, "
                                f"step:{step+1}/{steps}"
                                ": training loss {:.4f}, F1-mic {:.4f}, F1-mac {:.4f}".format(loss.item(), f1_mic, f1_mac)
                                )
                    logger.writeln(iter_msg)

            if cuda:
                print("current memory:",
                    torch.cuda.memory_allocated(device=batch_pred.device) / 1024 / 1024)

        if args.profile:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))

        # evaluate
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac = evaluate(
                model, g, feats, labels, val_mask, multitask)
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1-micro: {:.4f}'.format(best_f1))
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))
            val_msg = "Val F1-mic: {:.4f}, Val F1-mac: {:.4f}\n".format(val_f1_mic, val_f1_mac)
            val_msg += "-"*80
            logger.writeln(val_msg)

    end_time = time.time()
    print(f'training using time {end_time-start_time}')

    # test
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    test_f1_mic, test_f1_mac = evaluate(
        model, g, feats, labels, test_mask, multitask)
    test_msg = "Test F1-mic {:.4f}, Test F1-mac {:.4f}". format(test_f1_mic, test_f1_mac)
    logger.writeln(test_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="reddit-self-loop",
                        help="dataset name")
    parser.add_argument("--rootdir", type=str, default=get_download_dir(),
                        help="directory to read dataset from")
    # mmap features or not
    parser.add_argument("--feat-mmap", action='store_true', help="mmap dataset features")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="the frequency to save model")
    parser.add_argument("--batch-nodes", type=int, default=1000,
                        help="number of training nodes in a minibatch")
    parser.add_argument("--batch-clusters", type=int, default=20,
                        help="number of clusters sampled per round")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--fan-out", type=str, default="5,10,15")
    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")
    parser.add_argument("--profile", action='store_true',
                        help="Enable profiling of training process")

    args = parser.parse_args()

    main(args)

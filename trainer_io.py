import sys, os, argparse, time, random
import tqdm
from pyinstrument import Profiler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl

from modules import SAGE, SAGE_mlp, SAGE_res_incep, GAT, GAT_mlp, GIN
import gnnos
from graphloader import GnnosNodePropPredDataset
from sampler import GnnosIter

# def poll(post_queue):
#     while True:
#         tic = time.time()
#         num_nodes, batch_coo, batch_labels = post_queue.get()
#         print("polled coo, creating DGLGraph")
#         graph = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
#         graph.ndata['label'] = batch_labels
#         graph.create_formats_()
#         print(f"#graph: {graph}")
#         toc = time.time()
#         print(f"Iter Done: {toc-tic:.2f}s")

def train(args, in_feats, num_classes, data_queue):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    batch_size = args.bsize2

    if args.model == 'gat':
        model = GAT_mlp(in_feats, args.num_hidden, num_classes, args.n_layers, heads=4, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_feats, args.num_hidden, num_classes, num_layers=args.n_layers, dropout=args.dropout)
    else:
        model = SAGE(in_feats, args.num_hidden, num_classes, args.n_layers, F.relu, args.dropout)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=args.lr_decay, patience=args.lr_step, verbose=True)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)

    print("Training starts")
    # print("qsize:", data_queue.qsize())

    marker = time.time()
    while True:
        profiler = Profiler(interval=0.01)
        profiler.start()
        data = data_queue.get()
        tic = time.time()
        num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask = data
        graph = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
        train_nids = torch.nonzero(batch_train_mask, as_tuple=True)[0]
        graph.create_formats_()
        print(f"dgl graph creation: {time.time()-tic:.2f}s")
        profiler.stop()

        # graph.ndata['label'] = batch_labels
        # graph.ndata['feat'] = batch_feat

        dataloader = dgl.dataloading.DataLoader(
            graph,
            train_nids,
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            use_prefetch_thread=False, pin_prefetcher=False)
        iterator = enumerate(dataloader)

        profiler.start()
        recycle_factor = args.recycle
        model.train()
        batch_start = time.time()
        batch_iter = 0
        while batch_iter < len(dataloader) * recycle_factor:
            try:
                batch_iter += 1
                _, (input_nodes, output_nodes, blocks) = next(iterator)
                if batch_iter == 1:
                    first_minibatch = time.time()
                    print(f"mfg size:", len(input_nodes))
                if len(output_nodes) < 0.1 * args.bsize2:
                    # skip batches with too few training nodes
                    # print(f"skip {len(output_nodes)} nodes")
                    continue

                # Load the input features as well as output labels
                x = batch_feat[input_nodes].to(device).float()
                y = batch_labels[output_nodes].to(device).flatten().long()
                # label data is incorrect, use randint for now: doesn't affect computation
                y[:] = torch.randint(num_classes, y.shape, device=y.device)
                # x = blocks[0].srcdata['feat'].float()
                # y = blocks[-1].dstdata['label'].flatten().long()
                # Compute loss and prediction
                y_hat = model(blocks, x)
                loss = F.nll_loss(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step(loss)
                # train_acc = MF.accuracy(y_hat, y)

                # if (batch_iter+1) % args.log_every == 0:
                #     print(f"Iter {batch_iter+1}, train acc: {train_acc:.4f}")
            except StopIteration:
                if batch_iter < len(dataloader) * recycle_factor:
                    dataloader = dgl.dataloading.DataLoader(
                        graph,
                        train_nids,
                        sampler,
                        device=device,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False,
                        num_workers=args.num_workers,
                        use_prefetch_thread=False, pin_prefetcher=False)
                    iterator = enumerate(dataloader)
        batch_end = time.time()
        print(f"    mega-batch overall: {batch_end-batch_start:.2f}s")
        print(f"    mega-batch compute: {batch_end-first_minibatch:.2f}s")
        print(f"    num_iters={batch_iter}, num_workers={args.num_workers}")
        profiler.stop()
        profiler.print()
        print(f"############ mega-batch {time.time()-marker:.2f}s ############# \n")
        marker = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-papers100M')
    parser.add_argument("--model", type=str, default='sage')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--num-hidden', type=int, default=256,
                        help="Numer of hidden feature dimensions")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--lr-decay', type=float, default=0.9999,
                        help="Learning rate decay")
    parser.add_argument('--lr-step', type=int, default=1000,
                        help="Learning rate scheduler steps")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Number of sampling workers for host-gpu hierarchy")
    parser.add_argument("--psize", type=int, default=16384)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--bsize2", type=int, default=1024)
    parser.add_argument("--recycle", type=int, default=1,
                        help="Number of training passes over the host data before recycling")
    parser.add_argument("--rho", type=float, default=1,
                        help="recycling increasing factor")
    parser.add_argument("--io-threads", type=int, default=32,
                        help="threads to load data from storage (could be larger than #cpus)")
    parser.add_argument("--log-every", type=int, default=100,
                        help="number of steps of logging training acc/loss")
    args = parser.parse_args()
    args.fanout = list(map(int, args.fanout.split(',')))

    print(args)
    print("set NUM_THREADS to", args.io_threads)
    torch.set_num_threads(args.io_threads)
    gnnos.set_io_threads(args.io_threads)

    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize)
    in_feats = data.node_feat.metadata.shape[1]
    num_classes = data.num_classes

    it = iter(GnnosIter(data, args.bsize))

    context = mp.get_context('spawn')
    data_queue = context.Queue(maxsize=1)
    trainer = context.Process(target=train, args=(args, in_feats, num_classes, data_queue))
    trainer.start()

    iters = 0
    duration = []
    for i in range(args.n_epochs):
        print("Loading starts")
        tic = time.time()
        # profiler = Profiler(interval=0.01)
        # profiler.start()

        for data in it:
            num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask = data
            print(f"#nodes: {num_nodes}, #edges: {len(batch_coo[0])}, #train: {batch_train_mask.int().sum()}")
            print(f"batch_feat: {batch_feat.shape}")
            assert num_nodes == batch_feat.shape[0]
            assert num_nodes == batch_labels.shape[0]
            assert num_nodes == batch_train_mask.shape[0]
            del num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask
            data_queue.put(data)

        #     profiler.stop()
        #     profiler.print()
        #     profiler.start()
        # profiler.stop()
        toc = time.time()
        print(f"{len(it)} iters took {toc-tic:.2f}s")
        duration.append(toc-tic)

    print(f"On average: {np.mean(duration):.2f}")

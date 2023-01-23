import os, argparse, time
import tqdm, shutil

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics.functional as MF
import dgl

from model.gnn import SAGE, SAGE_mlp, SAGE_res_incep, GAT, GAT_mlp, GIN
import gnnos
from graphloader import BaselineNodePropPredDataset, GnnosNodePropPredDataset
from sampler_mp import GnnosIterShm
from partition_utils import MetisMinCutBalanced, MetisMinVolBalanced, RandomNodePartitioner
from logger import Logger

def train_serial(data, args, model, optimizer, lr_scheduler, sampler):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    batch_size = args.bsize2

    tic = time.time()
    num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask = data
    graph = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
    graph.create_formats_()
    # graph.ndata['feat'] = batch_feat
    # graph.ndata['label'] = batch_labels
    train_nids = torch.nonzero(batch_train_mask, as_tuple=True)[0]
    # print(f"[T] dgl graph creation: {time.time()-tic:.2f}s")

    dataloader = dgl.dataloading.DataLoader(
        graph,
        train_nids,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    iterator = enumerate(dataloader)

    recycle_factor = args.recycle
    model.train()
    batch_iter = 0
    while batch_iter < len(dataloader) * recycle_factor:
        try:
            batch_iter += 1
            _, (input_nodes, output_nodes, blocks) = next(iterator)
            if batch_iter == 1:
                first_minibatch = time.time()
                print(f"[T] MFG size:", len(input_nodes))
            if len(input_nodes) == len(output_nodes):
                # skip batches with only isolated nodes
                continue
            # Load the input features as well as output labels
            x = batch_feat[input_nodes].to(device).float()
            y = batch_labels[output_nodes].to(device).flatten().long()
            # Compute loss and prediction
            y_hat = model(blocks, x)
            loss = F.nll_loss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss)
            train_acc = MF.accuracy(y_hat.detach(), y).cpu()
            if (batch_iter+1) % args.log_every == 0:
                print(f"[T] Iter {batch_iter+1}, train acc: {train_acc:.4f}")
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
                    num_workers=args.num_workers)
                iterator = enumerate(dataloader)
    batch_end = time.time()
    print(f"[T] mega-batch overall: {batch_end-tic:.2f}s, compute: {batch_end-first_minibatch:.2f}s")
    return train_acc, batch_end-tic

def evaluate(args, model, eval_data):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    g = eval_data.graph
    g.ndata['feat'] = eval_data.node_feat
    g.ndata['label'] = eval_data.labels

    idx = eval_data.get_idx_split()
    val_nid = idx['valid']
    test_nid = idx['test']

    # validate with current graph
    valid_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.test_fanout)
    valid_dataloader = dgl.dataloading.DataLoader(
            g, val_nid, valid_sampler, device=device,
            batch_size=args.bsize3, shuffle=False, drop_last=False,
            num_workers=8)
    test_sampler = dgl.dataloading.MultiLayerNeighborSampler(args.test_fanout)
    test_dataloader = dgl.dataloading.DataLoader(
            g, test_nid, test_sampler, device=device,
            batch_size=args.bsize3, shuffle=False, drop_last=False,
            num_workers=8)

    model.eval()
    with torch.no_grad():
        ys = []
        y_hats = []
        #  for it, (_, _, blocks) in enumerate(tqdm.tqdm(valid_dataloader)):
        for it, (_, _, blocks) in enumerate(valid_dataloader):
            x = blocks[0].srcdata['feat'].float()
            ys.append(blocks[-1].dstdata['label'].flatten().long())
            y_hats.append(model(blocks, x))
        y_hats, ys = torch.cat(y_hats), torch.cat(ys)
        valid_acc = MF.accuracy(y_hats, ys).cpu()
        valid_loss = F.nll_loss(y_hats, ys).cpu()

        ys = []
        y_hats = []
        #  for it, (_, _, blocks) in enumerate(tqdm.tqdm(test_dataloader)):
        for it, (_, _, blocks) in enumerate(test_dataloader):
            x = blocks[0].srcdata['feat'].float()
            ys.append(blocks[-1].dstdata['label'].flatten().long())
            y_hats.append(model(blocks, x))
        y_hats, ys = torch.cat(y_hats), torch.cat(ys)
        test_acc = MF.accuracy(y_hats, ys).cpu()
        test_loss = F.nll_loss(y_hats, ys).cpu()

    return (valid_acc, valid_loss), (test_acc, test_loss)


def train(data_queue: mp.SimpleQueue, resp_queue: mp.SimpleQueue,
          buffers, in_feats, num_classes, args):
    '''
    trainer is spawned in another process
    receives commands and data via data_queue:
    '''
    if args.model == 'gat':
        if args.mlp:
            model = GAT_mlp(in_feats, args.num_hidden, num_classes, args.n_layers, heads=4, dropout=args.dropout)
        else:
            model = GAT(in_feats, args.num_hidden, num_classes, args.n_layers, heads=4, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_feats, args.num_hidden, num_classes, num_layers=args.n_layers, dropout=args.dropout)
    elif args.model == 'sage':
        if args.use_incep:
            model = SAGE_res_incep(in_feats, args.num_hidden, num_classes, args.n_layers, F.leaky_relu, args.dropout)
        elif args.mlp:
            model = SAGE_mlp(in_feats, args.num_hidden, num_classes, args.n_layers, F.relu, args.dropout)
        else:
            model = SAGE(in_feats, args.num_hidden, num_classes, args.n_layers, F.relu, args.dropout)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=args.lr_decay, patience=args.lr_step, verbose=True)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)

    print("[W] trainer starts")
    src_buffer, dst_buffer, label_buffer, feat_buffer = buffers
    eval_data = None

    # loop over training runs
    iter_duration = []
    trainer_duration = []
    train_acc = 0
    model.reset_parameters()
    while True:
        tic = time.time()
        data = data_queue.get()
        if data[0] == 'quit':
            break
        elif data[0] == 'reset':
            model.reset_parameters()
            resp_queue.put(('done', {
                'time-per-iter':        trainer_duration,
                'time-per-iter(all)':   iter_duration,
            }))
            iter_duration = []
            trainer_duration = []
            train_acc = 0
        elif data[0] == 'eval':
            if eval_data is None:
                eval_data = BaselineNodePropPredDataset(name=args.dataset, root=args.root, mmap_feat=False)
            val_result, test_result = evaluate(args, model, eval_data)
            resp_queue.put(('done', train_acc, val_result, test_result))
        elif data[0] == 'train':
            num_nodes, src_interval, dst_interval, label_interval, feat_interval, \
                batch_train_mask = data[1]
            batch_coo = (src_buffer[slice(*src_interval)], dst_buffer[slice(*dst_interval)])
            batch_labels = label_buffer[slice(*label_interval)]
            batch_feat = feat_buffer[slice(*feat_interval)]
            print(f"[W] Retrieve data: {time.time()-tic:.2f}s")
            print(f"[W] megabatch #nodes: {num_nodes}, #edges: {len(batch_coo[0])}, "
                f"#train: {batch_train_mask.int().sum()}")
            assert num_nodes == batch_feat.shape[0]
            assert num_nodes == batch_labels.shape[0]
            assert num_nodes == batch_train_mask.shape[0]
            train_acc, train_time = train_serial(
                (num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask),
                args, model, optimizer, lr_scheduler, sampler)
            # free as much memory as possible, otherwise memory requirements double
            del num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask
            del data
            src_buffer.free(src_interval)
            dst_buffer.free(dst_interval)
            label_buffer.free(label_interval)
            feat_buffer.free(feat_interval)

            trainer_duration.append(train_time)
            iter_duration.append(time.time()-tic)

        else:
            raise RuntimeError(f"Invalid msg: {data}")


if __name__ == "__main__":
    #  from pyinstrument import Profiler
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-products')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--model", type=str, default='sage')
    parser.add_argument("--mlp", action="store_true", help="add an MLP before outputs")
    parser.add_argument("--use-incep", action="store_true")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--num-hidden', type=int, default=256,
                        help="Numer of hidden feature dimensions")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument("--test-fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for evaluation")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--lr-decay', type=float, default=0.9999,
                        help="Learning rate decay")
    parser.add_argument('--lr-step', type=int, default=1000,
                        help="Learning rate scheduler steps")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of sampling workers for host-gpu hierarchy")
    parser.add_argument("--part", type=str, default="metis",
                        help="Partitioning method to use")
    parser.add_argument("--psize", type=int, default=4096,
                        help="#Partitions")
    parser.add_argument("--bsize", type=int, default=512,
                        help="#Partitions in a megabatch")
    parser.add_argument("--bsize2", type=int, default=1024,
                        help="Batch size used for GPU training")
    parser.add_argument("--bsize3", type=int, default=1024,
                        help="Batch size used during evaluation")
    parser.add_argument("--popular-ratio", type=float, default=0)
    parser.add_argument("--recycle", type=float, default=1,
                        help="Number of training passes over the host data before recycling")
    parser.add_argument("--rho", type=float, default=1,
                        help="recycling increasing factor")
    parser.add_argument("--io-threads", type=int, default=64,
                        help="threads to load data from storage (should be larger than #cpus)")
    parser.add_argument("--log-every", type=int, default=20,
                        help="number of steps of logging training acc/loss")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="evaluate every such epochs")
    parser.add_argument("--use-old-feat", action='store_true')
    parser.add_argument("--comment", type=str, help="Extra comments to print out to the trace")
    args = parser.parse_args()
    args.fanout = list(map(int, args.fanout.split(',')))
    args.test_fanout = list(map(int, args.test_fanout.split(',')))

    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print(f"\n## [{os.environ['HOSTNAME']}] {args.comment}")
    print(time_stamp)
    print(args)
    assert args.n_layers == len(args.fanout)
    assert args.n_layers == len(args.test_fanout)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    print(f"Training with GPU: {device}")
    print("set IO_THREADS to", args.io_threads)
    gnnos.set_io_threads(args.io_threads)

    # choose partitioner
    if args.part == "metis" or args.part == "metis-cut":
        partitioner = MetisMinCutBalanced()
    elif args.part == "metis-vol":
        partitioner = MetisMinVolBalanced()
    elif args.part == "rand":
        partitioner = RandomNodePartitioner()
    else:
        raise NotImplementedError
    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize,
        partitioner=partitioner, topk=args.popular_ratio, use_old_feat=args.use_old_feat)
    if args.use_old_feat:
        in_feats = data.node_feat.shape[1]
    else:
        in_feats = data.node_feat.metadata.shape[1]
    num_classes = data.num_classes

    ctx = mp.get_context('spawn')
    loader = GnnosIterShm(data, bsize=args.bsize, ctx=ctx)
    del data
    shm_buffers = loader.src_buffer, loader.dst_buffer, loader.label_buffer, loader.feat_buffer
    trainer = ctx.Process(target=train,
        args=(loader.data_queue, loader.resp_queue, shm_buffers, in_feats, num_classes, args))
    trainer.start()
    logger = Logger(args.runs, args)
    iter_per_epoch = args.psize // args.bsize

    for run in range(args.runs):
        for epoch in range(args.n_epochs):
            tic = time.time()
            for data in loader:
                # loader will push data into the data queue
                # print("Gnnos loader:", data[1], data[2], data[3], data[4])
                loader.train(data)
                print(f"Run {run}: epoch {epoch}, iter {loader.n}")
            if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
                accus = loader.evaluate()
                logger.add_result(run, accus)
                print(f"Epoch {epoch+1}: Val acc {accus[1]:.4f}, Test acc {accus[2]:.4f}")
        loader_stats, trainer_stats = loader.reset()
        print(f"Run {run}:\nloader:{loader_stats[:iter_per_epoch*2]}\n"
            f"trainer:{trainer_stats['time-per-iter(all)'][:iter_per_epoch*2]}")
        logger.print_statistics(run)
    loader.finish()
    logger.print_statistics()
    trainer.join()


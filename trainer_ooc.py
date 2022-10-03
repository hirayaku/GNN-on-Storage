import multiprocessing
import sys, os, argparse, time, random, gc
import tqdm
from pyinstrument import Profiler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl

from modules import SAGE

from graphloader import (
    split_tensor, PartitionMethod,
    PartitionSampler, PartitionedGraphLoader,
    HBatchGraphLoader)

from dataloader import HBatchDataLoader

def train_block_batching(model, opt, g, train_set, batch_size, num_workers, use_ddp=False, passes=1):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    sampler = PartitionSampler(train_set, device=device) # prefetch_ndata=['label', 'train_mask'])
    dataloader = dgl.dataloading.DataLoader(
            g, torch.arange(len(train_set)), sampler, device=device,
            batch_size=batch_size, shuffle=True, drop_last=False,
            use_ddp=use_ddp, num_workers=num_workers)

    model.train()
    t0 = time.time()
    for _ in range(passes):
        for it, (_, _, sg) in enumerate(dataloader):
            assert sg.device == device
            sg_train_mask = sg.ndata['train_mask']
            y_hat = model(sg, sg.ndata['feat'])[sg_train_mask]
            y = sg.ndata['label'][sg_train_mask]
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
        dist.barrier()

    tt = time.time()

    model.eval()
    ys = []
    y_hats = []
    for it, (_, _, sg) in enumerate(dataloader):
        with torch.no_grad():
            val_mask = sg.ndata['val_mask']
            ys.append(sg.ndata['label'][val_mask])
            y_hats.append(model(sg, sg.ndata['feat'])[val_mask])
            acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))

    tv = time.time()
    return acc, tt - t0, tv - tt

def train_ns_batching(model, opt, g, train_set, batch_size, fanout, num_workers, use_ddp=False, passes=1):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()
    gc.collect()
    torch.cuda.empty_cache() # prevent CUDA OOM?

    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #        fanout, prefetch_node_feats=['feat'], prefetch_labels=['label'])
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    train_dataloader = dgl.dataloading.DataLoader(
            g, train_set, sampler, device=device,
            batch_size=batch_size, shuffle=True, drop_last=False,
            use_ddp=use_ddp, num_workers=num_workers,
            use_prefetch_thread=False, pin_prefetcher=False)

    model.train()
    for _ in range(passes):
        for it, (_, _, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat'].float()
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print(f'Loss {loss.item():.5f} | Acc {acc.item():.4f} | GPU Mem {mem:.2f} MB')
        dist.barrier()

    return time.time() - t0

def eval_ns_batching(model, g, eval_set, batch_size, fanout, num_workers, use_ddp=False):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()
    gc.collect()
    torch.cuda.empty_cache() # prevent CUDA OOM?

    # validate with current graph
    eval_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(fanout))
    eval_dataloader = dgl.dataloading.DataLoader(
            g, eval_set, eval_sampler, device=device,
            batch_size=batch_size, shuffle=True, drop_last=False,
            use_ddp=use_ddp, num_workers=0,
            use_prefetch_thread=False, pin_prefetcher=False)

    model.eval()
    ys = []
    y_hats = []

    with torch.no_grad():
        for it, (_, _, blocks) in enumerate(eval_dataloader):
            x = blocks[0].srcdata['feat'].float()
            ys.append(blocks[-1].dstdata['label'].long())
            y_hats.append(model(blocks, x))
    acc = MF.accuracy(torch.cat(y_hats), torch.cat(ys))
    tv = time.time()
    return acc, tv - t0

def train_gpu_ddp(rank, world_size, feature_dim, num_classes,
        subgraph_queue: multiprocessing.Queue, resp_queue, args):

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    assert rank == dist.get_rank()
    print(f"Trainer {rank}: init_process_group, world size = {world_size}")

    device = torch.cuda.current_device()
    model = SAGE(feature_dim[0], args.num_hidden, num_classes, args.n_layers, F.relu, 0.5).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    #  opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    print(f"Trainer {rank}: model created on {device}, micro batchsize={args.bsize2}")

    # select sampling method
    #  if args.sampling == "NS":
    #      model.module.forward = model.module.forward_mfg
    #      fanout = [int(f) for f in args.fanout.split(',')]
    #      train_gpu = lambda g, train_set: train_ns_batching(model, opt, g, train_set,
    #          args.bsize2, fanout, args.num_workers, use_ddp=True, passes=args.recycle)
    #  elif args.sampling == "Block":
    #      model.module.forward = model.module.forward_full
    #      train_gpu = lambda g, train_set: train_block_batching(model, opt, g, train_set,
    #          args.bsize2, args.num_workers, use_ddp=True, passes=args.recycle)
    assert args.sampling == "NS"

    mega_batch = 0
    while True:
        start = time.time()
        if rank == 0:
            print(f"Trainer got MegaBatch {mega_batch}")
            profiler = Profiler()
            profiler.start()

        epoch = mega_batch // ((args.psize-1)//args.bsize+1)
        mega_batch += 1
        msg = subgraph_queue.get()
        if msg is None:
            break

        adj, train_mask, val_mask, labels, features, intervals = msg
        g = dgl.graph(('csc', adj))
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['label'] = labels
        g.ndata['feat'] = features

        if args.sampling == "NS":
            train_set = g.nodes()[train_mask]
        else:
            train_set = split_tensor(g.nodes(), intervals)

        fanout = [int(f) for f in args.fanout.split(',')]
        tt = train_ns_batching(model, opt, g, train_set, args.bsize2, fanout,
                args.num_workers, use_ddp=True, passes=args.recycle)

        if epoch % args.eval_every == 0 and epoch != 0:
            val_set = g.nodes()[val_mask]
            val_acc, tv = eval_ns_batching(model, g, val_set, args.bsize3,
                    fanout, args.num_workers, use_ddp=True)
            val_acc = val_acc / world_size
            dist.reduce(val_acc, 0)
            if rank == 0:
                end = time.time()
                print(f"Val acc: {val_acc:.4f} | MegaBatch time: {end-start:.2f}s, train: {tt:.2f}s, val: {tv:.2f}s")
        else:
            if rank == 0:
                end = time.time()
                print(f"MegaBatch time: {end-start:.2f}s, train: {tt:.2f}s")

        if rank == 0:
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))
        dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HieBatching DDP Trainer',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="ogbn-products", help="Dataset")
    parser.add_argument("--root", type=str, default="datasets", help="Dataset location")
    parser.add_argument("--tmpdir", type=str, default=".",
                        help="Location to save intermediate data (prefer fast storage)")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--num-hidden', type=int, default=256,
                        help="Numer of hidden feature dimensions")
    parser.add_argument('--lr', type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of DDP processes")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument("--psize", type=int, default=256,
                        help="Number of node partitions")
    parser.add_argument("--bsize", type=int, default=16,
                        help="Batch size for storage-host hierarchy")
    parser.add_argument("--bsize2", type=int, default=4,
                        help="Batch size for host-gpu hierarchy")
    parser.add_argument("--bsize3", type=int, default=128,
                        help="Batch size used during evaluation")
    parser.add_argument("--recycle", type=int, default=0,
                        help="Number of training passes over the host data before recycling")
    parser.add_argument("--sampling", type=str, default="NS",
                        help="Choose sampling method for host-gpu hierarchy (NS | Block)")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling method for host-gpu hierarchy")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of graph sampling workers for host-gpu hierarchy")
    args = parser.parse_args()
    n_procs = args.n_procs
    n_epochs = args.n_epochs
    if args.recycle == 0:
        args.recycle = n_procs

    print(args)
    print(f"Storage-Host batching: {(args.psize, args.bsize)}")
    if args.sampling == "NS":
        print(f"Host-GPU batching: NeighborSampling {args.bsize2}, [{args.fanout}]")
    else:
        print(f"Host-GPU batching: BlockSampling {(args.bsize, args.bsize2)}")

    import gnnos
    gnnos.verbose()
    gnnos.set_tmp_dir(args.tmpdir)

    gloader = HBatchGraphLoader(name=args.dataset, root=args.root, p_size=args.psize)
    #  gloader = HBatchGraphLoader(name=args.dataset, root=args.root, p_size=args.psize,
    #          p_method=PartitionMethod.EXTERNAL)
    dataloader = HBatchDataLoader(gloader, batch_size=args.bsize)

    import torch.multiprocessing as mp
    context = mp.get_context("spawn")

    queues = []
    trainers = []

    resp_queue = context.Queue(maxsize=1)
    for rank in range(n_procs):
        q = context.Queue(maxsize=1)
        w = context.Process(
            target=train_gpu_ddp,
            args=(rank, n_procs, gloader.feature_dim(), gloader.num_classes(), q, resp_queue, args))
        w.start()

        queues.append(q)
        trainers.append(w)

    mega_batch = 0
    for ep in range(n_epochs):
        loop_start = time.time()
        for i, (sg, features, intervals) in enumerate(dataloader):
            # NB: be careful, don't send DGLGraph directly over the queue, otherwise confusing bugs
            # or even segfaults will emerge. DGLGraph seems not 100% compatible with multiprocessing
            adj = sg.adj_sparse('csc')
            train_mask, val_mask = sg.ndata['train_mask'], sg.ndata['val_mask']
            num_train = torch.nonzero(train_mask).shape[0]
            labels = sg.ndata['label'].flatten().long()
            print(f"Load batch: {time.time()-loop_start:.2f}s")

            for q in queues:
                q.put((adj, train_mask, val_mask, labels, features, intervals))
            print(f"Loader sent MegaBatch {mega_batch} "
                f"(num_nodes={sg.num_nodes()}, num_edges={sg.num_edges()}, num_train={num_train})")
            mega_batch += 1
            del sg
            loop_start = time.time()

        print(f"{'='*10} Epoch {ep} Loaded {'='*10}" )

    for q in queues:
        q.put(None)
    for w in trainers:
        w.join()

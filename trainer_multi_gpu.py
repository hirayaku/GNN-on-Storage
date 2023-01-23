import os, time, random
from pyinstrument import Profiler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torch.multiprocessing as mp
import torchmetrics.functional as MF
import dgl

from model.gnn import SAGE

def train_ddp(data_queue: mp.SimpleQueue, resp_queue: mp.SimpleQueue,
        buffers, in_feats, num_classes, args, rank=0, world_size=1):

    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    print(f"Trainer {rank}: init_process_group, world size = {world_size}")

    device = torch.cuda.current_device()
    model = SAGE(in_feats, args.num_hidden, num_classes, args.n_layers, F.relu, args.dropout).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Trainer {rank}: model created on {device}, batchsize={args.bsize2}")

    profiler = Profiler()

    mega_batch = 0
    while True:
        start = time.time()
        if rank == 0:
            print(f"Trainer got MegaBatch {mega_batch}")
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

        train_set = g.nodes()[train_mask]
        train_acc, train_time = train_serial(g, args, model, opt, lr_scheduler, sampler)
        # tt = train_serial(model, opt, g, train_set, args.bsize2, args.fanout,
        #         args.num_workers, use_ddp=True, passes=args.recycle)

        if rank == 0:
            end = time.time()
            print(f"MegaBatch time: {end-start:.2f}s, train: {tt:.2f}s")
            profiler.stop()
            print(profiler.output_text(unicode=True, color=True))

        dist.barrier()

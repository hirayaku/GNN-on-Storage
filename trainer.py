import multiprocessing
import sys, os, argparse, time, random
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
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

    def inference(self, g, mask, device, batch_size=1000, num_workers=0, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        if buffer_device is None:
            buffer_device = device

        # TODO: can't hold all the hidden features in memory
        # one approach is to create a partitioning from valid sets.
        # Validation is performed on randomly selected partitions
        for l, layer in enumerate(self.layers[:-1]):
            y = torch.zeros(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device)
            for _, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata['h']
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            g.ndata['h'] = y

        # inference on the last layer: apply masks to save computation
        target_nodes = g.nodes()[mask]
        dataloader_ll = dgl.dataloading.DataLoader(
                g, target_nodes.to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        layer_l = self.layers[-1]
        y = torch.zeros(target_nodes.shape, self.n_classes, device=buffer_device)
        for _, output_nodes, blocks in tqdm.tqdm(dataloader_ll):
            x = blocks[0].srcdata['h']
            h = layer_l(blocks[0], x)
            y[output_nodes] = h.to(buffer_device)

        return y

from graphloader import (
    split_tensor,
    PartitionSampler, PartitionedGraphLoader,
    HBatchGraphLoader)

from dataloader import PartitionDataLoader, HBatchDataLoader

def train(model, opt, g, train_set,batch_size, num_workers=0, use_ddp=False, passes=1):
    # TODO: try prefetch, try uva
    sampler = PartitionSampler(train_set) # prefetch_ndata=['label', 'train_mask'])
    dataloader = dgl.dataloading.DataLoader(
            g, torch.arange(len(train_set)), sampler,
            batch_size=batch_size, shuffle=True, drop_last=False,
            use_ddp=use_ddp, num_workers=num_workers)

    model.train()
    t0 = time.time()
    for _ in range(passes):
        for it, (sg, _, _) in enumerate(dataloader):
            sg_train_mask = sg.ndata['train_mask']
            y_hat = model(sg, sg.ndata['feat'])[sg_train_mask]
            y = sg.ndata['label'][sg_train_mask].flatten()
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 10 == 0:
                acc = MF.accuracy(y_hat, y)
                mem = torch.cuda.max_memory_allocated() / 1000000
                print('Loss', loss.item(), 'Acc', acc.item(), 'GPU Mem', mem, 'MB')
    
    # validate with subgraph
    model.eval()
    val_tp = 0
    val_total = 0
    for it, (sg, _, _) in enumerate(dataloader):
        val_mask = sg.ndata['val_mask']
        y_hat = model(sg, sg.ndata['feat'])[val_mask]
        pred = torch.argmax(y_hat, dim=1)
        truth = sg.ndata['label'][val_mask].flatten().long()
        val_tp += (pred == truth).sum().item()
        val_total += pred.shape[0]

    tt = time.time()
    return val_tp, val_total, tt - t0

def train_ddp(rank, world_size, subgraph_queue: multiprocessing.Queue, feature_dim, num_classes):
    # TODO: enable cuda
    # torch.cuda.set_device(rank)
    # dist.init_process_group('nccl', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    dist.init_process_group('gloo', 'tcp://127.0.0.1:12347', world_size=world_size, rank=rank)
    assert rank == dist.get_rank()
    print(f"Trainer {rank}: init_process_group, world size = {world_size}")

    model = SAGE(feature_dim[0], 256, num_classes)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
    model = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    print(f"Trainer {rank}: model created")

    batch_size = 4
    num_workers = 0

    while True:
        # print(f"Trainer {rank}: queue size = {subgraph_queue.qsize()}")
        msg = subgraph_queue.get()
        if msg is None:
            break

        adj, train_mask, val_mask, labels, features, intervals = msg
        g = dgl.graph(('csr', adj))
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask 
        g.ndata['label'] = labels
        g.ndata['feat'] = features
        partitions = split_tensor(g.nodes(), intervals)
        val_tp, val_total, t = train(model, opt, g, partitions, batch_size, num_workers,
            use_ddp=True, passes=world_size*2)
        
        print(f"Val acc: {val_tp/val_total*100:.2f}%")
        print(f"Step time: {t:.2f}s")

        # TODO: validate and test
        # if rank == 0:
        #     model.eval()
        #     # dataloader for validation pass
        #     neighbor_sampler = dgl.dataloading.NeighborSampler(
        #         [15, 10, 5], prefetch_labels=['label'])
        #     valid_dataloader = dgl.dataloading.DataLoader(
        #         gloader.graph, gloader.valid_idx(), neighbor_sampler, batch_size=1024, shuffle=True,
        #         drop_last=False, num_workers=0)
        #     print('Validation acc:', acc.item())
        dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HieBatching DDP Trainer')
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of DDP processes")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of training epochs")
    args = parser.parse_args()
    n_procs = args.n_procs
    n_epochs = args.n_epochs

    # from viztracer import VizTracer
    # tracer = VizTracer(output_file=f"traces/ddp-proc{n_procs}.json", min_duration=10)
    # tracer.start()

    # gloader = PartitionedGraphLoader(1024, overwrite=False,
    #     name='ogbn-products', root='/mnt/md0/graphs', mmap=True)
    # gloader.formats(['coo', 'csc'])
    # dataloader = PartitionDataLoader(gloader, batch_size=128)

    import gnnos
    gnnos.verbose()

    gloader = HBatchGraphLoader(name=args.dataset, root="/mnt/md0/inputs", p_size=256)
    dataloader = HBatchDataLoader(gloader, batch_size=8)

    import torch.multiprocessing as mp
    context = mp.get_context("spawn")

    queues = []
    trainers = []

    for rank in range(n_procs):
        q = context.Queue(maxsize=1)
        w = context.Process(
            target=train_ddp,
            args=(rank, n_procs, q, gloader.feature_dim(), gloader.num_classes()))
        w.start()
        
        queues.append(q)
        trainers.append(w)

    for ep in range(n_epochs):
        print(f"{'='*10} Epoch {ep} {'='*10}" )
        loop_start = time.time()
        for i, (sg, features, intervals) in enumerate(dataloader):
            # NB: be careful, don't send DGLGraph directly over the queue, otherwise confusing bugs
            # or even segfaults will emerge. DGLGraph seems not 100% compatible with multiprocessing
            adj = sg.adj_sparse('csr')
            train_mask, val_mask = sg.ndata['train_mask'], sg.ndata['val_mask']
            labels = sg.ndata['label']

            print(f"load: {time.time()-loop_start:.2f}s")
            for q in queues:
                q.put((adj, train_mask, val_mask, labels, features, intervals))
            print("Put subgraph (num_nodes={}, num_edges={}) into queue".format(
                sg.num_nodes(), sg.num_edges() ))
            del sg
            loop_start = time.time()

    for q in queues:
        q.put(None)
    for w in trainers:
        w.join()

    # tracer.stop()
    # tracer.save()

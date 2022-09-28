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
from graphloader import BaselineNodePropPredDataset, GnnosNodePropPredDataset
from sampler import GnnosIter

def poll(post_queue):
    while True:
        tic = time.time()
        num_nodes, batch_coo, batch_labels = post_queue.get()
        print("polled coo, creating DGLGraph")
        graph = dgl.graph(('coo', batch_coo), num_nodes=num_nodes)
        graph.ndata['label'] = batch_labels
        graph.create_formats_()
        print(f"#graph: {graph}")
        toc = time.time()
        print(f"Iter Done: {toc-tic:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='samplers + trainers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='ogbn-papers100M')
    parser.add_argument("--root", type=str, default=os.path.join(os.environ['DATASETS'], 'gnnos'))
    parser.add_argument("--psize", type=int, default=16384)
    parser.add_argument("--bsize", type=int, default=1024)
    parser.add_argument("--io-threads", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    print(args)
    print("set NUM_THREADS to", args.io_threads)
    torch.set_num_threads(args.io_threads)
    gnnos.set_io_threads(args.io_threads)

    data = GnnosNodePropPredDataset(name=args.dataset, root=args.root, psize=args.psize)
    data.num_nodes, data.parts
    tuple(data.graph.coo_part, data.graph.coo_src, data.graph.coo_dst)
    tuple(data.scache.coo_part, data.scache.coo_src, data.scache.coo_dst)
    data.scache_nids, data.scache_feat
    data.labels, data.node_feat

    # it = iter(GnnosIter(data, args.bsize))

    # context = mp.get_context('forkserver')
    # post_q = mp.Queue(maxsize=1)
    # poller = context.Process(target=poll, args=(post_q,))
    # poller.start()

    # duration = []
    # for i in range(args.epochs):
    #     print("Loading starts")
    #     tic = time.time()
    #     profiler = Profiler(interval=0.01)
    #     profiler.start()
    #     for num_nodes, batch_coo, batch_labels, batch_feat, batch_train_mask in it:
    #         print(f"#nodes: {num_nodes}, #edges: {len(batch_coo[0])}, batch_feat: {batch_feat.shape}")
    #         assert num_nodes == batch_feat.shape[0]
    #         assert num_nodes == batch_labels.shape[0]
    #         assert num_nodes == batch_train_mask.shape[0]
    #         post_q.put((num_nodes, batch_coo))
    #         profiler.stop()
    #         profiler.print()
    #         profiler.start()
    #     profiler.stop()
    #     toc = time.time()
    #     print(f"{len(it)} iters took {toc-tic:.2f}s")
    #     duration.append(toc-tic)
    # print(f"On average: {np.mean(duration):.2f}")


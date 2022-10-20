import multiprocessing
import sys, os, argparse, time, random
import shutil
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.optim
import torchmetrics.functional as MF
import dgl

from modules import SAGE, SAGE_mlp, SAGE_res_incep, GAT, GAT_mlp, GIN
from graphloader import BaselineNodePropPredDataset
from partition_utils import MetisMinCutBalanced, MetisMinVolBalanced, RandomNodePartitioner
import sampler as HBSampler
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

def eval_ns_batching(model, g, eval_set, batch_size, fanout, num_workers, use_ddp=False):
    '''
    run evaluation on eval_set, using neighbor sampling of fanout
    requires full access on the dataset
    '''
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()

    # validate with current graph
    eval_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    #  eval_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(fanout))
    eval_dataloader = dgl.dataloading.DataLoader(
            g, eval_set, eval_sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            use_ddp=use_ddp, num_workers=num_workers,
            use_prefetch_thread=False, pin_prefetcher=False)

    model.eval()
    ys = []
    y_hats = []

    with torch.no_grad():
        for it, (_, _, blocks) in enumerate(eval_dataloader):
            x = blocks[0].srcdata['feat'].float()
            ys.append(blocks[-1].dstdata['label'].flatten().long())
            y_hats.append(model(blocks, x))
    y_hats, ys = torch.cat(y_hats), torch.cat(ys)
    acc = MF.accuracy(y_hats, ys)
    loss = F.nll_loss(y_hats, ys)
    tv = time.time()
    return acc, loss, tv - t0

# TODO
def eval_hier_batching(model, cluster_iter, eval_name, batch_size, fanout, num_workers):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()

    megabatches = tqdm.tqdm(enumerate(cluster_iter)) if args.progress else enumerate(cluster_iter)
    eval_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)

    model.eval()
    ys = []
    y_hats = []

    with torch.no_grad():
        for j, (subgraph, _) in megabatches:
            eval_mask = subgraph.ndata[eval_name] & (~subgraph.ndata['cache_mask'])
            eval_nids = subgraph.nodes()[eval_mask]
            if j == 0:
                print(f'#nodes:{subgraph.num_nodes()}, #edges:{subgraph.num_edges()}')
                print(f'#eval: {eval_nids.shape[0]}')

            dataloader = dgl.dataloading.DataLoader(
                subgraph, eval_nids, eval_sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers,
                use_prefetch_thread=False, pin_prefetcher=False)

            for  _, _, blocks in dataloader:
                x = blocks[0].srcdata['feat'].float()
                ys.append(blocks[-1].dstdata['label'].flatten().long())
                y_hats.append(model(blocks, x))

        y_hats, ys = torch.cat(y_hats), torch.cat(ys)
        acc = MF.accuracy(y_hats, ys)
        loss = F.nll_loss(y_hats, ys)
        tv = time.time()
        return acc, loss, tv - t0

# return: (highest_train, highest_valid, final_train, final_test)
def train(args, data, partitioner, tb_writer):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    g = data.graph
    idx = data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    n_classes = data.num_classes
    in_feats = g.ndata['feat'].shape[1]

    cluster_iterator = HBSampler.ClusterIterV2(args.dataset, g, args.psize, args.bsize, args.hsize,
            partitioner=partitioner, sample_topk=args.popular_sample, popular_ratio=args.popular_ratio)

    if args.model == 'gat':
        if args.mlp:
            model = GAT_mlp(in_feats, args.num_hidden, n_classes, args.n_layers, heads=4, dropout=args.dropout)
        else:
            model = GAT(in_feats, args.num_hidden, n_classes, args.n_layers, heads=4, dropout=args.dropout)
    elif args.model == 'gin':
        model = GIN(in_feats, args.num_hidden, n_classes, num_layers=args.n_layers, dropout=args.dropout)
    elif args.model == 'sage':
        if args.use_incep:
            model = SAGE_res_incep(in_feats, args.num_hidden, n_classes, args.n_layers, F.leaky_relu, args.dropout)
        elif args.mlp:
            model = SAGE_mlp(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
        else:
            model = SAGE(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=args.lr_decay, patience=args.lr_step, verbose=True)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        global_iter = 0
        model.reset_parameters()
        for epoch in range(args.n_epochs):
            epoch_iter = 0
            # recycle between [1, 5) iterations
            recycle_factor = min(max(args.recycle * args.rho**epoch, 1), 5)
            print(f"Epoch {epoch+1}/{args.n_epochs}, Recycle: {recycle_factor:.2f}")
            model.train()
            megabatches = tqdm.tqdm(enumerate(cluster_iterator)) if args.progress \
                else enumerate(cluster_iterator)
            for j, (subgraph, train_nids) in megabatches:
                sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
                dataloader = dgl.dataloading.DataLoader(
                    subgraph,
                    train_nids,
                    sampler,
                    device=device,
                    batch_size=args.bsize2,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers,
                    use_prefetch_thread=False, pin_prefetcher=False)
                iterator = enumerate(dataloader)

                batch_start = time.time()
                batch_iter = 0
                while batch_iter < len(dataloader) * recycle_factor:
                    try:
                        batch_iter += 1
                        _, (input_nodes, output_nodes, blocks) = next(iterator)
                        if batch_iter == 1:
                            first_minibatch = time.time()
                        # skip batches with too few training nodes
                        if len(output_nodes) < 0.1 * args.bsize2:
                            print(f"skip {len(output_nodes)} nodes")
                            continue
                        # Load the input features as well as output labels
                        x = blocks[0].srcdata['feat'].float()
                        y = blocks[-1].dstdata['label'].flatten().long()

                        # Compute loss and prediction
                        y_hat = model(blocks, x)
                        loss = F.nll_loss(y_hat, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step(loss)

                        global_iter += 1
                        epoch_iter += 1
                        train_acc = MF.accuracy(y_hat, y)
                        if run == 0:
                            tb_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_iter)
                            tb_writer.add_scalar('Train/loss', loss.item(), global_iter)
                            tb_writer.add_scalar('Train/accu', train_acc, global_iter)
                            tb_writer.add_scalar('Train/recycle', recycle_factor, global_iter)

                        if (epoch_iter+1) % args.log_every == 0:
                            print(f"Epoch {epoch+1}/{args.n_epochs}, Iter {epoch_iter+1}",
                                f"train acc: {train_acc:.4f}, nodes: {subgraph.num_nodes()}")
                    except StopIteration:
                        if batch_iter < len(dataloader) * recycle_factor:
                            dataloader = dgl.dataloading.DataLoader(
                                subgraph,
                                train_nids,
                                sampler,
                                device=device,
                                batch_size=args.bsize2,
                                shuffle=True,
                                drop_last=False,
                                num_workers=args.num_workers,
                                use_prefetch_thread=False, pin_prefetcher=False)
                            iterator = enumerate(dataloader)
                batch_end = time.time()
                # print(f"    mega-batch overall: {batch_end-batch_start:.2f}s")
                # print(f"    mega-batch compute: {batch_end-first_minibatch:.2f}s")
                # print(f"    num_iters={batch_iter}, num_workers={args.num_workers}")

            if (epoch + 1) % args.eval_every == 0:
                train_acc, _, _ = eval_ns_batching(model, g, train_nid, args.bsize2,
                        args.fanout, args.num_workers, use_ddp=False)
                val_acc, val_loss, _ = eval_ns_batching(model, g, val_nid, args.bsize2,
                        args.fanout, args.num_workers, use_ddp=False)
                test_acc, test_loss, _ = eval_ns_batching(model, g, test_nid, args.bsize3,
                        args.test_fanout, args.num_workers, use_ddp=False)
                if run == 0:
                    tb_writer.add_scalar('Valid/accu', val_acc, epoch)
                    tb_writer.add_scalar('Valid/loss', val_loss, epoch)
                    tb_writer.add_scalar('test/accu', test_acc, epoch)
                    tb_writer.add_scalar('test/loss', test_loss, epoch)
                logger.add_result(run, (train_acc, val_acc, test_acc))
                print(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")
        logger.print_statistics(run)
    logger.print_statistics()

    return logger.get_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HieBatching DDP Trainer',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--dataset", type=str, default="ogbn-products",
                        help="Dataset")
    parser.add_argument("--root", type=str, default=f"{os.environ['DATASETS']}/gnnos",
                        help="Dataset location")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--model", type=str, default="sage", help="GNN model (sage|gat|gin)")
    parser.add_argument("--mlp", action="store_true", help="add an MLP before outputs")
    parser.add_argument("--use-incep", action="store_true")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument('--num-hidden', type=int, default=256,
                        help="Numer of hidden feature dimensions")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument("--test-fanout", type=str, default="20,20,20",
                        help="Choose sampling fanout for host-gpu hierarchy (test)")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta_1 in Adam optimizer')
    parser.add_argument('--lr-decay', type=float, default=0.9999,
                        help="Learning rate decay")
    parser.add_argument('--lr-step', type=int, default=1000,
                        help="Learning rate scheduler steps")
    parser.add_argument('--wt-decay', type=float, default=0,
                        help="Model parameter weight decay")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of DDP processes")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--part", type=str, default="metis",
                        help="Partitioning method to use")
    parser.add_argument("--psize", type=int, default=256,
                        help="Number of node partitions")
    parser.add_argument("--bsize", type=int, default=16,
                        help="Batch size for storage-host hierarchy")
    parser.add_argument("--hsize", type=int, default=16,
                        help="Helper partition size")
    parser.add_argument("--bsize2", type=int, default=1024,
                        help="Batch size for host-gpu hierarchy")
    parser.add_argument("--bsize3", type=int, default=64,
                        help="Batch size used during evaluation")
    parser.add_argument("--recycle", type=int, default=1,
                        help="Number of training passes over the host data before recycling")
    parser.add_argument("--rho", type=float, default=1,
                        help="recycling increasing factor")
    parser.add_argument("--log-every", type=int, default=10,
                        help="number of steps of logging training acc/loss")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of graph sampling workers for host-gpu hierarchy")
    parser.add_argument("--popular-ratio", type=float, default=0)
    parser.add_argument("--popular-sample", action="store_true")
    parser.add_argument("--progress", action="store_true", help="show training progress bar")
    parser.add_argument("--comment", type=str, help="Extra comments to print out to the trace")
    args = parser.parse_args()
    n_procs = args.n_procs
    n_epochs = args.n_epochs

    seed = random.randint(0,1024**3)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)

    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print(f"\n## [{os.environ['HOSTNAME']}] {args.comment} seed:{seed}")
    print(time_stamp)
    print(args)
    print(f"Storage-Host batching: {(args.psize, args.bsize)}")
    print(f"Host-GPU batching: NeighborSampling {args.bsize2}, [{args.fanout}]")
    args.fanout = list(map(int, args.fanout.split(',')))
    args.test_fanout = list(map(int, args.test_fanout.split(',')))
    assert args.n_layers == len(args.fanout)
    assert args.n_layers == len(args.test_fanout)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    print(f"Training with GPU: {device}")

    # choose model
    model = args.model
    if args.model == 'sage':
        if args.use_incep:
            model = "sage-ri"
        elif args.mlp:
            model = "sage-mlp+bn"
    # choose partitioner
    partition = args.part
    if args.part == "metis" or args.part == "metis-cut":
        partitioner = MetisMinCutBalanced()
    elif args.part == "metis-vol":
        partitioner = MetisMinVolBalanced()
    elif args.part == "rand":
        partitioner = RandomNodePartitioner()
    else:
        raise NotImplementedError

    # load dataset and statistics
    data = BaselineNodePropPredDataset(name=args.dataset, root=args.root, mmap_feat=False)
    g = data.graph
    g.ndata['feat'] = data.node_feat
    g.ndata['label'] = data.labels
    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    in_feats = g.ndata['feat'].shape[1]

    idx = data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    g.ndata['train_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['train_mask'][train_nid] = True
    g.ndata['valid_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['valid_mask'][val_nid] = True
    g.ndata['test_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['test_mask'][test_nid] = True
    n_train_samples = len(train_nid)
    n_val_samples = len(val_nid)
    n_test_samples = len(test_nid)

    print(f"""----Data statistics------
    #Nodes {n_nodes}
    #Edges {n_edges}
    #Classes/Labels (multi binary labels) {n_classes}
    #Train samples {n_train_samples}
    #Val samples {n_val_samples}
    #Test samples {n_test_samples}
    #Labels     {g.ndata['label'].shape}
    #Features   {g.ndata['feat'].shape}"""
    )

    if args.popular_sample:
        popular = 'sample'
    else:
        popular = 'fixed'
    log_path = f"log/{args.dataset}/HB-{model}-{partition}-{popular}-c{args.popular_ratio}" \
            + f"-r{args.recycle}*{args.rho}-p{args.psize}-b{args.bsize}-{time_stamp}"
    tb_writer = SummaryWriter(log_path, flush_secs=5)

    try:
        accu = train(args, data, partitioner, tb_writer)
    except:
        shutil.rmtree(log_path)
        print("** removed tensorboard log dir **")
        raise

    tb_writer.add_hparams({
        'seed': seed,'model': model,'num_hidden': args.num_hidden, 'fanout': str(args.fanout),
        'use_incep': args.use_incep, 'mlp': args.mlp, 'lr': args.lr, 'lr-decay': args.lr_decay,
        'dropout': args.dropout, 'weight-decay': args.wt_decay,
        'partition': args.part, 'psize': args.psize, 'bsize': args.bsize, 'bsize2': args.bsize2,
        'rho': args.rho, 'recycle': args.recycle, 'popular_ratio': args.popular_ratio, 'popular_method': popular,
        },
        {'hparam/val_acc': accu[0].item(), 'hparam/test_acc': accu[1].item() }
        )
    tb_writer.close()

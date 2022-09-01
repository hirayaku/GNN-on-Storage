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

from modules import SAGE, SAGE_mlp, SAGE_res_incep
from graphloader import GraphLoader, PartitionMethod, PartitionSampler, PartitionedGraphLoader
from dataloader import PartitionDataLoader
import sampler as HBSampler
#  from sampler import ClusterIterV2
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

def eval_ns_batching(model, g, eval_set, batch_size, fanout, num_workers, use_ddp=False):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()

    # validate with current graph
    eval_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    #  eval_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(len(fanout))
    eval_dataloader = dgl.dataloading.DataLoader(
            g, eval_set, eval_sampler, device=device,
            batch_size=batch_size, shuffle=True, drop_last=False,
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

# return: (highest_train, highest_valid, final_train, final_test)
def train(args, tb_writer):
    assert args.sampling == "NS"

    data = GraphLoader(name=args.dataset, root=args.root, mmap=False)
    g = data.graph
    g.ndata['feat'] = data.node_features
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask'] if 'val_mask' in g.ndata else g.ndata['valid_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']

    train_nid = g.nodes()[train_mask]
    val_nid = g.nodes()[val_mask]
    test_nid = g.nodes()[test_mask]

    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes()
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    print("#Labels shape:", g.ndata['label'].shape)
    print("#Features shape:", g.ndata['feat'].shape)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    if args.part == "metis":
        partitioner = HBSampler.MetisBalancedPartitioner()
    elif args.part == "rand":
        partitioner = HBSampler.RandomNodePartitioner()
    else:
        raise NotImplementedError
    cluster_iterator = HBSampler.ClusterIterV2(args.dataset, g, args.psize, args.bsize, args.hsize,
            partitioner=partitioner, sample_topk=not args.popular_sample, popular_ratio=args.popular_ratio)

    if args.use_incep:
        model = SAGE_res_incep(in_feats, args.num_hidden, n_classes, args.n_layers, F.leaky_relu, args.dropout)
    elif args.mlp:
        model = SAGE_mlp(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    else:
        model = SAGE(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, 0.999), weight_decay=args.wt_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=args.lr_decay,
                                                              patience=1000, verbose=True)

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
            for j, (subgraph, train_nids) in enumerate(cluster_iterator):
                if j == 0 or j == len(cluster_iterator)-1:
                    print(f'#nodes:{subgraph.num_nodes()}, #edges:{subgraph.num_edges()}')
                    print(f'#train:{train_nids.shape[0]}')

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

                batch_iter = 0
                while batch_iter < len(dataloader) * recycle_factor:
                    try:
                        batch_iter += 1
                        step, (_, _, blocks) = next(iterator)
                    #  for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
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

                        if (epoch_iter+1) % args.log_every == 0:
                            print(f"Epoch {epoch+1}/{args.n_epochs}, Iter {epoch_iter+1}",
                                f"train acc: {train_acc:.4f}, nodes: {subgraph.num_nodes()}")
                    except StopIteration:
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

            if (epoch + 1) % args.eval_every == 0:
                train_acc, _, _ = eval_ns_batching(model, g, train_nid, args.bsize2,
                        args.fanout, args.num_workers, use_ddp=False)
                val_acc, val_loss, _ = eval_ns_batching(model, g, val_nid, args.bsize2,
                        args.fanout, args.num_workers, use_ddp=False)
                test_acc, test_loss, _ = eval_ns_batching(model, g, test_nid, args.bsize3,
                        args.test_fanout, args.num_workers, use_ddp=False)
                if run == 0:
                    tb_writer.add_scalar('Valid/accu', val_acc, global_iter)
                    tb_writer.add_scalar('Valid/loss', val_loss, global_iter)
                    tb_writer.add_scalar('test/accu', test_acc, global_iter)
                    tb_writer.add_scalar('test/loss', test_loss, global_iter)
                logger.add_result(run, (train_acc, val_acc, test_acc))
                print(f"Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")
                print(f"Best val: {logger.best_val:.4f}")
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
    parser.add_argument("--root", type=str, default=f"{os.environ['DATASETS']}/baseline",
                        help="Dataset location")
    parser.add_argument("--runs", type=int, default=1)
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
    parser.add_argument("--sampling", type=str, default="NS",
                        help="Choose sampling method for host-gpu hierarchy (NS | Block)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="number of steps of logging training acc/loss")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of graph sampling workers for host-gpu hierarchy")
    parser.add_argument("--mlp", action="store_true", help="add an MLP before outputs")
    parser.add_argument("--use-incep", action="store_true")
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
    if args.sampling == "NS":
        print(f"Host-GPU batching: NeighborSampling {args.bsize2}, [{args.fanout}]")
    else:
        print(f"Host-GPU batching: BlockSampling {(args.bsize, args.bsize2)}")
    args.fanout = list(map(int, args.fanout.split(',')))
    args.test_fanout = list(map(int, args.test_fanout.split(',')))
    assert args.n_layers == len(args.fanout)
    assert args.n_layers == len(args.test_fanout)
    assert args.part in ["metis", "rand", "deg-bucket"]
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    print(f"Training with GPU: {device}")

    model = "plain"
    if args.use_incep:
        model = "incep"
    elif args.mlp:
        model = "mlp+bn"
    partition = args.part
    if args.popular_sample:
        popular = 'sample'
    else:
        popular = 'fixed'
    log_path = f"log/{args.dataset}/{model}-{partition}-{popular}-c{args.popular_ratio}" \
            + f"-r{args.recycle}*{args.rho}-p{args.psize}-h{args.hsize}-b{args.bsize}-{time_stamp}"
    tb_writer = SummaryWriter(log_path, flush_secs=5)
    try:
        accu = train(args, tb_writer)
    except:
        shutil.rmtree(log_path)
        print("** removed tensorboard log dir **")
        raise
    tb_writer.add_hparams({
        'model': model,'num_hidden': args.num_hidden, 'fanout': str(args.fanout),
        'lr': args.lr, 'dropout': args.dropout, 'rho': args.rho, 'recycle': args.recycle,
        'partition': args.part, 'psize': args.psize, 'hsize': args.hsize, 'bsize': args.bsize, 'bsize2': args.bsize2,
        'seed': seed, 'popular_ratio': args.popular_ratio, 'popular_nodes': popular
        },
        {'hparam/val_acc': accu[0].item(), 'hparam/test_acc': accu[1].item() }
        )
    tb_writer.close()

    '''
    print("Step", step)
    # 0-hop nodes
    nodes = output_nodes.cpu()
    deg_sum, deg2_sum = 0, 0
    for node in nodes:
        adjs = subgraph.in_edges(node)[0]
        deg = adjs.shape
        node_g = subgraph.ndata[dgl.NID][node]
        adjs2 = g.in_edges(node_g)[0]
        deg2 = adjs2.shape
        deg_sum += deg[0]
        deg2_sum += deg2[0]
        if deg[0] > deg2[0]:
            raise Exception
    print(f"0-hop: {nodes.shape[0]}")
    print(f"avg_deg={deg_sum/nodes.shape[0]:.2f}, {deg_sum/deg2_sum*100:.2f}%")
    # 1-hop nodes
    nodes = blocks[0].srcnodes()
    nodes = blocks[0].srcdata[dgl.NID][nodes].cpu()
    deg_sum, deg2_sum = 0, 0
    for node in nodes:
        adjs = subgraph.in_edges(node)[0]
        deg = adjs.shape
        node_g = subgraph.ndata[dgl.NID][node]
        adjs2 = g.in_edges(node_g)[0]
        deg2 = adjs2.shape
        deg_sum += deg[0]
        deg2_sum += deg2[0]
        if deg[0] > deg2[0]:
            raise Exception
    print(f"1-hop: {nodes.shape[0]}")
    print(f"avg_deg={deg_sum/nodes.shape[0]:.2f}, {deg_sum/deg2_sum*100:.2f}%")
    '''

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
from graphloader import BaselineNodePropPredDataset
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

def eval_ns_batching(model, g, eval_set, batch_size, fanout, num_workers, use_ddp=False):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    t0 = time.time()

    # validate with current graph
    eval_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    eval_dataloader = dgl.dataloading.DataLoader(
            g, eval_set, eval_sampler, device=device,
            batch_size=batch_size, shuffle=True, drop_last=False,
            use_ddp=use_ddp, num_workers=num_workers,
            use_prefetch_thread=False, pin_prefetcher=False)

    model.eval()
    ys = []
    y_hats = []

    with torch.no_grad():
        minibatches = tqdm.tqdm(enumerate(eval_dataloader)) if args.progress else enumerate(eval_dataloader)
        for it, (_, _, blocks) in minibatches:
            x = blocks[0].srcdata['feat'].float()
            ys.append(blocks[-1].dstdata['label'].flatten().long())
            y_hats.append(model(blocks, x))
    y_hats, ys = torch.cat(y_hats), torch.cat(ys)
    acc = MF.accuracy(y_hats, ys)
    loss = F.nll_loss(y_hats, ys)
    tv = time.time()
    return acc, loss, tv - t0

# return: (highest_train, highest_valid, final_train, final_test)
def train(args, data, tb_writer):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    g = data.graph
    g.ndata['feat'] = data.node_feat
    g.ndata['label'] = data.labels
    idx = data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    n_classes = data.num_classes
    in_feats = node_feat.shape[1]

    if args.use_incep:
        model = SAGE_res_incep(in_feats, args.num_hidden, n_classes, args.n_layers, F.leaky_relu, args.dropout)
    elif args.mlp:
        model = SAGE_mlp(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    else:
        model = SAGE(in_feats, args.num_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=args.lr_decay, patience=args.patience, verbose=True)

    logger = Logger(args.runs, args)
    for run in range(args.runs):
        global_iter = 0
        model.reset_parameters()

        sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
        dataloader = dgl.dataloading.DataLoader(
            g,
            train_nid,
            sampler,
            device=device,
            batch_size=args.bsize2,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            use_prefetch_thread=False, pin_prefetcher=False)

        for epoch in range(args.n_epochs):
            epoch_iter = 0
            print(f"Epoch {epoch+1}/{args.n_epochs}")
            minibatches = tqdm.tqdm(enumerate(dataloader)) if args.progress else enumerate(dataloader)
            model.train()
            for step, (input_nodes, output_nodes, blocks) in minibatches:
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

                if args.progress:
                    minibatches.set_postfix({'acc': train_acc.item(), 'loss': loss.item()})
                elif (epoch_iter+1) % args.log_every == 0:
                    print(f"Epoch {epoch+1}/{args.n_epochs}, Iter {epoch_iter+1} train acc: {train_acc:.4f}")

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
                print(f"Val acc: {val_acc:.4f}")
                print(f"Test acc: {test_acc:.4f}")
                logger.add_result(run, (train_acc, val_acc, test_acc))
        logger.print_statistics(run)
    logger.print_statistics()

    return logger.get_result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Baseline minibatch-based GNN training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--dataset", type=str, default="ogbn-products",
                        help="Dataset name")
    parser.add_argument("--root", type=str, default=f"{os.environ['DATASETS']}/gnnos",
                        help="Dataset location")
    parser.add_argument("--logdir", type=str, default="log",
                        help="Log location")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="Number of hidden feature dimensions")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument("--test-fanout", type=str, default="20,20,20",
                        help="Choose sampling fanout for host-gpu hierarchy (test)")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.9999,
                        help="Learning rate decay")
    parser.add_argument('--wt-decay', type=float, default=0,
                        help="Model parameter weight decay")
    parser.add_argument("--patience", type=float, default=1000,
                        help="Learning rate decay patience")
    parser.add_argument("--n-procs", type=int, default=1,
                        help="Number of DDP processes")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--bsize2", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--bsize3", type=int, default=128,
                        help="Batch size used during evaluation")
    parser.add_argument("--log-every", type=int, default=10,
                        help="number of steps of logging training acc/loss")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of graph sampling workers for host-gpu hierarchy")
    parser.add_argument("--mlp", action="store_true", help="add an MLP before outputs")
    parser.add_argument("--use-incep", action="store_true", help="use SAGE_res_incep model")
    parser.add_argument("--progress", action="store_true", help="show training progress bar")
    parser.add_argument("--comment", type=str, help="Extra comments to print out to the trace")
    args = parser.parse_args()

    n_procs = args.n_procs
    n_epochs = args.n_epochs

    # remember the random seed
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
    print(f"\n## {args.comment}, seed: {seed}")
    print(time_stamp)
    print(args)
    args.fanout = list(map(int, args.fanout.split(',')))
    args.test_fanout = list(map(int, args.test_fanout.split(',')))
    assert args.n_layers == len(args.fanout)
    assert args.n_layers == len(args.test_fanout)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    print(f"Training with GPU: {device}")

    model = "plain"
    if args.use_incep:
        model = "incep"
    elif args.mlp:
        model = "mlp+bn"

    # load dataset and statistics
    data = BaselineNodePropPredDataset(name=args.dataset, root=args.root, mmap_feat=False)
    g = data.graph
    node_feat = data.node_feat
    labels = data.labels

    idx = data.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    n_train_samples = len(train_nid)
    n_val_samples = len(val_nid)
    n_test_samples = len(test_nid)

    n_classes = data.num_classes
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    in_feats = node_feat.shape[1]

    print(f"""----Data statistics------'
    #Nodes {n_nodes}
    #Edges {n_edges}
    #Classes/Labels (multi binary labels) {n_classes}
    #Train samples {n_train_samples}
    #Val samples {n_val_samples}
    #Test samples {n_test_samples}
    #Labels     {labels.shape}
    #Features   {node_feat.shape}"""
    )

    log_path = f"{args.logdir}/{args.dataset}/sage-{model}-{time_stamp}"
    tb_writer = SummaryWriter(log_path)

    try:
        accu = train(args, data, tb_writer)
    except:
        print("** Unable to finish training due to error, removing tensorboard log **")
        shutil.rmtree(log_path)
        raise

    tb_writer.add_hparams({
        'seed': seed,'model': model,'num_hidden': args.num_hidden, 'fanout': str(args.fanout),
        'use_incep': args.use_incep, 'mlp': args.mlp, 'lr': args.lr, 'dropout': args.dropout, 'weight-decay': args.wt_decay,
        'bsize2': args.bsize2,
        },
        {'hparam/val_acc': accu[0].item(), 'hparam/test_acc': accu[1].item() }
        )
    tb_writer.close()


import sys, os, argparse, time, random
import os.path as osp
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

import modules
from torch.utils.tensorboard import SummaryWriter
import sampler, utils, partition_utils

# TODO: divide par_li into two lists: the former one contains all and only training nodes
def partition(dataset, g, psize, method, partition_dir=".", cache_partition=True):
    print(f"Getting partitions of {dataset}...")
    if method.startswith("METIS"):
        partition_func = lambda g, psize: \
            partition_utils.get_partition_list(g, psize, g.ndata['train_mask'].int())
        print("METIS node partitioning")
    elif method.startswith("NEV"):
        if "RandSeed" in method:
            partition_func = partition_utils.get_nev_partition_list_randseed
            print("NEV partitioning method (random seeds)")
        else:
            partition_func = partition_utils.get_nev_partition_list
            print("NEV partitioning method (clustered seeds)")
    else:
        if "RandSeed" in method:
            partition_func = partition_utils.get_rand_partition_list
            print("random node partitioning (random seeds)")
        else:
            partition_func = partition_utils.get_rand_partition_list_clusterseed
            print("random node partitioning (clustered seeds)")

    # cache the partitions of known dataset&partition number
    if cache_partition:
        fn = osp.join(partition_dir, f'{dataset}_{method}_p{psize}.npy')
        if osp.exists(fn):
            par_li = np.load(fn, allow_pickle=True)
            par_li = [utils.to_torch_tensor(par) for par in par_li]
        else:
            os.makedirs(partition_dir, exist_ok=True)
            par_li = partition_func(g, psize)
            np.save(fn, [par.numpy() for par in par_li])
    else:
        par_li = partition_func(g, psize)
    
    return par_li

# evaluate model on g (full-batch) and calculate accuracy with different masks
def evaluate(device, model, g, feat, labels, masks, multitask=False):
    loss_f = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        logits = model.inference(g, feat)
    
    def calc_acc_loss(mask):
        logits_masked = logits[mask]
        labels_masked = labels[mask]
        f1_mic, f1_mac = utils.calc_f1(labels_masked.cpu().numpy(),
            logits_masked.cpu().numpy(), multitask)
        return f1_mic, f1_mac, loss_f(logits_masked, labels_masked)
    
    return [calc_acc_loss(mask) for mask in masks]

# evaluate model on g (in batches, on MFG) and calculate accuracy with different masks
def evaluate_mfg(device, model: nn.Module, g, feat, labels, masks, multitask=False):
    results = []
    loss_f = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for mask in masks:
            nodes = torch.nonzero(mask, as_tuple=True)[0]
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=0) # TODO: nonzero num_workers brings huge slowdown, why?
            logits = []
            for input_nodes, _, blocks in tqdm.tqdm(dataloader):
                logits.append(model(blocks, feat[input_nodes]))
            logits = torch.cat(logits).cpu().numpy()
            labels_masked = labels[mask].cpu().numpy()
            f1_mic, f1_mac = utils.calc_f1(labels_masked, logits, multitask)
            results.append((f1_mic, f1_mac, loss_f(logits, labels_masked)))
    return results

def main(args):
    # set rand seed: https://stackoverflow.com/a/5012617
    rnd_seed = random.randrange(sys.maxsize)
    random.seed(rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    data = utils.load_data(args)
    g = data.g
    feats = data.features
    labels = g.ndata['label']
    train_mask, val_mask, test_mask = \
        g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

    in_feats = feats.shape[1]
    n_classes = data.num_classes
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes %d
    #InFeats %d
    #Train samples %d (%.2f)
    #Val samples   %d (%.2f)
    #Test samples  %d (%.2f)""" %
            (n_nodes, n_edges, n_classes, in_feats,
            n_train_samples, n_train_samples / n_nodes * 100,
            n_val_samples, n_val_samples / n_nodes * 100,
            n_test_samples, n_test_samples / n_nodes * 100))

    # get partitions of the dataset
    dataset_dir = osp.join(args.rootdir, args.dataset.replace('-', '_'))
    partition_dir = osp.join(dataset_dir, "partitions")
    par_list = partition(args.dataset, g, args.psize, args.partition, partition_dir=partition_dir)

    subgraph_loader = sampler.ClusterIter(g, args.psize, args.isize, par_list, args.replacement)

    if args.profile_mfg:
        mfg_sizes = torch.zeros((args.n_epochs, n_train_samples, args.n_layers))
        # for each in-memory partitions, record #nodes, #edges, #training-nodes
        minibatch_stats = torch.zeros((args.n_epochs, args.psize // args.isize, 3))
        mfg_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
        print(f"Collecting MFG sizes under {args.partition} (p={args.psize}, i={args.isize})")
        for epoch in range(args.n_epochs):
            nid = 0
            for j, subgraph in enumerate(tqdm.tqdm(subgraph_loader)):
                batch_train_nodes = subgraph.nodes()[subgraph.ndata['train_mask']]
                minibatch_stats[epoch, j, :] = torch.tensor(
                    [subgraph.num_nodes(), subgraph.num_edges(), batch_train_nodes.size()[0]])
                for node in batch_train_nodes:
                    _, _, blocks = mfg_sampler.sample_blocks(subgraph, node)
                    mfg_sizes[epoch, nid, :] = torch.tensor([block.num_src_nodes() for block in blocks])
                    nid += 1
        stat_fn = osp.join(partition_dir, f"mfg_{args.partition}_{args.replacement}_p{args.psize}_i{args.isize}.npz") 
        np.savez(stat_fn, minibatch=minibatch_stats, mfg=mfg_sizes)
        print(f"Average input sizes of MFGs: {torch.mean(mfg_sizes, dim=1)}")
        print(f"Results written to {stat_fn}")
        return

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        # g = g.to(args.gpu)

    # create GraphSAGE model
    layer = modules.GraphSAGELayer
    model = modules.GNNModule(
                layer,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    if cuda:
        model.cuda()

    # Loss function
    if multitask:
        print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss(reduction="none")
    else:
        print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss(reduction="none")

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # use tensorboard to log training data
    logger = SummaryWriter(comment=f' {args.dataset} {args.partition} p={args.psize} i={args.isize} r={args.replacement} s={args.sampler}')
    log_dir = logger.log_dir

    start_time = time.time()
    best_f1 = -1
    device = val_mask.device

    for epoch in range(args.n_epochs):
        total_loss = 0
        for j, subgraph in enumerate(tqdm.tqdm(subgraph_loader)):
            batch_labels = subgraph.ndata['label']
            batch_train_mask = subgraph.ndata['train_mask']
            batch_train_nodes = subgraph.nodes()[batch_train_mask]

            model.train()
            # TODO: support different in-memory samplers
            # Full-Batch, move the entire graph to device
            subgraph = subgraph.to(device)
            subgraph_h = subgraph.ndata['feat']
            # forward
            batch_pred = model(subgraph, subgraph_h)
            loss = loss_f(batch_pred[batch_train_mask],
                          batch_labels[batch_train_mask])
            total_loss += loss.sum().item()
            loss = torch.mean(loss)
            # if we allow training nodes to appear more than once in different batches, the following is necessary
            # loss = torch.mean(loss / cluster_g.ndata['count'][batch_train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar("Train/loss-minibatch", loss, epoch * len(subgraph_loader) + j)

        logger.add_scalar("Train/loss", total_loss / n_train_samples, epoch)

        if epoch % args.val_every == args.val_every - 1:
            (val_f1_mic, _, val_loss), (test_f1_mic, _, test_loss) = evaluate(
                device, model, g, utils.to_torch_tensor(feats), labels, (val_mask, test_mask))
            print("Val F1-micro {:.4f}". format(val_f1_mic))
            print("Test F1-micro {:.4f}". format(test_f1_mic))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1-micro: {:.4f}'.format(best_f1))
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/F1-micro", val_f1_mic, epoch)
            logger.add_scalar("Test/loss", test_loss, epoch)
            logger.add_scalar("Test/F1-micro", test_f1_mic, epoch)

    end_time = time.time()
    print(f'training using time {end_time-start_time}')

    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pkl')))
    (test_f1_mic, _, test_loss), = evaluate(
        device, model, g, utils.to_torch_tensor(feats), labels, (test_mask,))
    print("Test F1-micro {:.4f}".format(test_f1_mic))

    logger.add_hparams(
        {"psize": args.psize, "isize": args.isize,
         "layers": args.n_layers, "hidden": args.n_hidden,
         "dropout": args.dropout, "epochs": args.n_epochs,
         "lr": args.lr, "weight-decay": args.weight_decay,
         "partition-method": args.partition,
         "replacement": args.replacement, "sampler": args.sampler,
         "sampler-bsize": 0 if args.sampler == "FB" else args.bsize,
         "rnd-seed": rnd_seed},
        {"test accuracy": test_f1_mic,
         "test loss": test_loss }, run_name=".")
    logger.close()

# def train(model: nn.Module, loss_f: nn.Module, optimizer: torch.optim.Optimizer,
#     subgraph_loader: object, sampler: object, batch_size: int):


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GNN training on storage')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dataset", type=str, default="ogbn-products",
                        help="dataset name")
    parser.add_argument("--rootdir", type=str, default=dgl.data.utils.get_download_dir(),
                        help="directory to read dataset from")
    parser.add_argument("--feat-mmap", action='store_true',
                        help="load node features in a separate file via mmap")

    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=0,
                        help="Weight for L2 loss")

    parser.add_argument("--psize", type=int, default=1600,
                        help="partition number")
    parser.add_argument("--isize", type=int, default=20,
                        help="in-memory partition number")
    parser.add_argument("--partition", type=str, default="METIS",
                        help="clustering method: METIS, NEV or RAND")
    parser.add_argument("--balance-ts", action='store_true',
                        help="balance the training set in multiple partitions")
    parser.add_argument("--replacement", type=str, default="ClusterGCN",
                        help="replacement policy for in-memory partitions: ClusterGCN")
    # minibatch sampler: NeighborSampling (NS), RandomWalk (RW), FullBatch (FB)
    parser.add_argument("--sampler", type=str, default="FB",
                        help="method to sample minibatches from in-memory partitions")
    parser.add_argument("--bsize", type=int, default=400,
                        help="batch size for minibatch sampler")
    parser.add_argument("--fanout", type=str, default="10,20,30",
                        help="fanout to use during neighbor sampling")

    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--profile-mfg", action="store_true",
                        help="profile the sizes of mfg")

    args = parser.parse_args()

    print(args)

    main(args)

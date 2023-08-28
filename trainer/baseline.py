import os, argparse, time, random, warnings
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from pyinstrument import Profiler

from models.pyg import gen_model
from data.graphloader import BaselineNodePropPredDataset
from datapipe.sampler_fn import PygNeighborSampler
from datapipe.custom_pipes import IterDataPipe
from datapipe.sampler_pipes import NodeSamplerDataPipe
from trainer import train, eval_full, eval_batch
from trainer.recorder import Recorder
import data.partitioner as pt

def edge_cuts(data, pt_assigns):
    src, dst, _ = data.adj_t.coo()
    return (pt_assigns[src]-pt_assigns[dst] != 0).int().sum().item()

def make_blocks(data: Data, nodes, num_blocks, mode):
    assigns = None
    if mode == "metis":
        assigns = pt.MetisWeightedPartitioner(data, num_blocks, nodes).partition()
    elif mode == "random":
        assigns = pt.RandomNodePartitioner(data, num_blocks).partition()
    elif mode == "fennel":
        # prepare labels to balance
        train_mask = index_to_mask(nodes, data.size(0))
        train_labels = data.y.clone().flatten().int()
        train_labels[~train_mask] = -1
        assigns = pt.ReFennelPartitioner(
            data, num_blocks, runs=5, slack=2, alpha_ratio=0.2,
            base=pt.FennelStrataPartitioner, labels=train_labels,
        ).partition()
    return pt.scatter_append(0, assigns[nodes], nodes, return_sequence=True), assigns

def make_ns_dp(
    data: Data, node_dp: IterDataPipe,
    fanout: list[int], **kwargs
):
    num_workers = kwargs.get('num_workers', 0)
    sampler = PygNeighborSampler(fanout) if fanout else None
    sampler_dp = NodeSamplerDataPipe(data, node_dp, sample_fn=sampler)
    return sampler_dp
    # rs = MultiProcessingReadingService(num_workers=num_workers)
    # return DataLoader2(sampler_dp, reading_service=rs)

def run(data, train_dp, val_dp, test_dp, args, logger):
    data.y = data.y.flatten()

    model = gen_model(in_feats, n_classes, args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer, factor=args.lr_decay, patience=args.lr_step, verbose=True)
    for run in range(args.runs):
        train_time = 0
        model.reset_parameters()
        logger.set_run(run)
        for epoch in range(args.num_epochs):
            print(f'>>> Epoch {epoch}')
            tic = time.time()
            loss, acc, *_ = train(model, optimizer, train_dp, args.device)
            toc = time.time()
            train_time += toc-tic
            print('Epoch Time (s): {:.4f}, Train acc: {:.4f}'.format(toc-tic, acc))
            logger.add(epoch, data={
                'train': {'loss': loss, 'acc': acc},
            })
            if (epoch + 1) % args.eval_every == 0:
                val_loss, val_acc = eval_batch(model, val_dp, args.device)
                # test_loss, test_acc = eval_batch(model, test_dp, args.device)
                test_loss, test_acc = 0, 0
                print(f"Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")
                logger.add(epoch, data={
                    'val': {'loss': val_loss, 'acc': val_acc},
                    'test': {'loss': test_loss, 'acc': test_acc},
                })
        print('Run {}: mean epoch time: {:.3f}'.format(run, train_time/args.num_epochs))
        print(logger.get_acc())
        logger.save(f'acc_study')

    return logger

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Baseline minibatch-based GNN training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv",
                        help="Dataset name")
    parser.add_argument("--root", type=str, default='/opt/datasets/new',
                        help="Dataset location")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mmap", action="store_true", help="use mmap")
    parser.add_argument("--model", type=str, default="sage", help="GNN model (sage|gat|gin)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="Number of hidden feature dimensions")
    parser.add_argument("--use-incep", action='store_true')
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument("--test-fanout", type=str, default="20,20,20",
                        help="Choose sampling fanout for host-gpu hierarchy (test)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9999,
                        help="Learning rate decay")
    parser.add_argument('--lr-step', type=int, default=1000,
                        help="Learning rate scheduler steps")
    parser.add_argument('--wt-decay', type=float, default=0,
                        help="Model parameter weight decay")
    parser.add_argument("--num-procs", type=int, default=1,
                        help="Number of DDP processes")
    parser.add_argument("--num-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--log-every", type=int, default=10,
                        help="number of steps of logging training acc/loss")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of graph sampling workers for host-gpu hierarchy")
    parser.add_argument("--hb", type=str, default='fennel',
                        help="hierachical batching of nodes")
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--block-ratio", type=int, default=8)
    args = parser.parse_args()

    args.seed = random.randint(0,1024**3)
    pyg.seed_everything(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    time_stamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    print(f"\n## [{os.environ.get('HOSTNAME', 'localhost')}], {time_stamp}")
    print(args)

    args.fanout = list(map(int, args.fanout.split(',')))
    args.test_fanout = list(map(int, args.test_fanout.split(',')))
    assert args.num_layers == len(args.fanout)
    assert args.num_layers == len(args.test_fanout)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        print(f"Training with GPU: {args.device}")
    else:
        args.device = torch.device('cpu')
        print(f"Training with CPU")

    dataset = BaselineNodePropPredDataset(
        name=args.dataset,
        root=args.root,
        mmap=args.mmap
    )
    assert not dataset.is_directed, "expect the dataset to be undirected"
    data = dataset[0]
    in_feats = data.x.shape[1]
    n_classes = dataset.num_classes
    idx = dataset.get_idx_split()
    train_nid = idx['train']
    val_nid = idx['valid']
    test_nid = idx['test']
    n_train_samples = len(train_nid)
    n_val_samples = len(val_nid)
    n_test_samples = len(test_nid)
    # shuffle train_nid
    train_nid = train_nid[torch.randperm(len(train_nid))]

    print(f"""----Data statistics------'
    #Nodes {data.num_nodes}
    #Edges {data.num_edges}
    #Classes/Labels (multi binary labels) {n_classes}
    #Train samples  {n_train_samples}
    #Val samples    {n_val_samples}
    #Test samples   {n_test_samples}
    #Labels     {data.y.shape}
    #Features   {data.x.shape}\n"""
    )

    from torch_geometric.sampler.utils import to_csc
    from torch_sparse import SparseTensor
    colptr, row, _ = to_csc(data)
    data.adj_t = SparseTensor(rowptr=colptr, col=row, sparse_sizes=data.size())

    from datapipe.batcher import global_batching_dp, hier_batching_dp
    if args.hb in ('random', 'metis', 'fennel'):
        train_blocks, assigns = make_blocks(data, train_nid, args.num_blocks, mode=args.hb)
        train_nid_dp = hier_batching_dp(
            train_blocks,
            args.num_blocks//args.block_ratio,
            args.batch_size,
            shuffle=True,
            drop_thres=1/4,
        )
        print(f"edge_cuts = {edge_cuts(data, assigns)}/{data.num_edges}")
        print(f"{len(train_blocks)} blocks, {len(train_blocks)//args.block_ratio} per batch")
        # XXX: check train_nid_dp yields the correct results, remove later
        if train_nid.size(0) < 1e6:
            shuffled = torch.cat(list(train_nid_dp))
            train_ratio = shuffled.size(0) / train_nid.size(0)
            print(f"utilized training nodes: {round(train_ratio*100, 2)}%")
            setattr(args, 'train_ratio', train_ratio)
            if train_ratio == 1:
                assert (shuffled.sort()[0] == train_nid.sort()[0]).all()
    elif args.hb == 'once':
        train_nid_dp = global_batching_dp(train_nid, args.batch_size, shuffle=False)
    elif args.hb is None:
        train_nid_dp = global_batching_dp(train_nid, args.batch_size, shuffle=True)
    train_dp = make_ns_dp(data, train_nid_dp, args.fanout, shuffle=True, batch_size=args.batch_size)
    val_nid_dp = global_batching_dp(val_nid, batch_size=args.batch_size, shuffle=False)
    val_dp = make_ns_dp(data, val_nid_dp, args.test_fanout)
    test_nid_dp = global_batching_dp(test_nid, batch_size=args.batch_size, shuffle=False)
    test_dp = make_ns_dp(data, test_nid_dp, args.test_fanout)

    logger = Recorder(info=args)
    try:
        run(data, train_dp, val_dp, test_dp, args, logger)
    except:
        if (logger._run > 0):
            ans = input("keep serialized logger data? [Y/n]: ")
            if not ans.upper().startswith('Y'):
                os.remove(f"acc_study/{args.dataset}_{logger.md5}.pkl")


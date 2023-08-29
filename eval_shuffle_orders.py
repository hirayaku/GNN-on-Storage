'''
This script is used to experiment with different shuffling orders of training nodes
It doesn't apply hierarchical batching of the graph topology yet.
'''
import os, argparse, time, random, warnings
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

from models.pyg import gen_model
from data.graphloader import NodePropPredDataset
from data.ops import scatter_append
from datapipe.sampler_fn import PygNeighborSampler
from datapipe.custom_pipes import IterDataPipe, LiteIterableWrapper
from trainer.helpers import train, eval_full
from trainer.recorder import Recorder

import data.partitioner as pt
from utils import emd

def edge_cuts(data, pt_assigns):
    src, dst, _ = data.adj_t.coo()
    return (pt_assigns[src]-pt_assigns[dst] != 0).int().sum().item()

class FennelStrataDegOrderPartitioner(pt.FennelStrataPartitioner):
    def __init__(self, g, psize, name='Fennel-strata-deg', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices

def make_blocks(data: Data, nodes, num_blocks, mode):
    if mode == "metis":
        assigns = pt.MetisWeightedPartitioner(data, num_blocks, nodes).partition()
    elif mode == "random":
        assigns = pt.RandomNodePartitioner(data, num_blocks).partition()
    elif mode == "fennel":
        # prepare labels to balance
        train_mask = index_to_mask(nodes, data.size(0))
        train_labels = data.y.clone().flatten().int()
        num_labels = train_labels.max().item()
        train_labels[~train_mask] = num_labels
        assigns = pt.ReFennelPartitioner(
            data, num_blocks, runs=4, slack=1.25,
            base=FennelStrataDegOrderPartitioner,
            labels=train_labels,
        ).partition()
    scattered, offsets, _ = scatter_append(0, assigns, nodes, num_blocks)
    blocked = [ scattered[offsets[i]:offsets[i+1]] for i in range(num_blocks) ]
    return blocked, assigns

def make_ns_dp(data: Data, node_dp: IterDataPipe, fanout: list[int]):
    sampler = PygNeighborSampler(fanout) if fanout else None
    data_dp = LiteIterableWrapper([data]).repeats(1000_000)
    data_dp = data_dp.zip(node_dp).map(sampler)
    return data_dp

def run(data, train_dp, val_nid, test_nid, args):
    recorder = Recorder(info=args)
    data.y = data.y.flatten()

    model = gen_model(in_feats, n_classes, args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wt_decay)
    for run in range(args.runs):
        train_time = 0
        model.reset_parameters()
        recorder.set_run(run)
        for epoch in range(args.num_epochs):
            print(f'>>> Epoch {epoch}')
            tic = time.time()
            loss, acc, *_ = train(model, optimizer, train_dp, args.device)
            toc = time.time()
            train_time += toc-tic
            print('Epoch Time (s): {:.4f}, Train acc: {:.4f}'.format(toc-tic, acc))
            recorder.add(epoch, data={
                'train': {'loss': loss, 'acc': acc},
            })
            if (epoch + 1) % args.eval_every == 0:
                results = eval_full(model, data, device=args.device, masks=(val_nid, test_nid))
                val_loss, val_acc = results[0]
                test_loss, test_acc = results[1]
                print(f"Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f}")
                recorder.add(epoch, data={
                    'val': {'loss': val_loss, 'acc': val_acc},
                    'test': {'loss': test_loss, 'acc': test_acc},
                })
        print('Run {}: mean epoch time: {:.3f}'.format(run, train_time/args.num_epochs))
        print(recorder.current_acc())
        # recorder.save(f'acc_study')

    return recorder

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Baseline minibatch-based GNN training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv",
                        help="Dataset name")
    parser.add_argument("--root", type=str, default='/mnt/md0/hb_datasets',
                        help="Dataset location")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--mmap", action="store_true", help="use mmap")
    parser.add_argument("--model", type=str, default="sage", help="GNN model (sage|gat|gin)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--num-hidden", type=int, default=256,
                        help="Number of hidden feature dimensions")
    parser.add_argument("--fanout", type=str, default="15,10,5",
                        help="Choose sampling fanout for host-gpu hierarchy")
    parser.add_argument("--test-fanout", type=str, default="20,20,20",
                        help="Choose sampling fanout for host-gpu hierarchy (test)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
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

    root = os.path.join(args.root, args.dataset.replace('-', '_'))
    dataset = NodePropPredDataset(root, mmap=args.mmap)
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

    from datapipe.batcher import hier_batching_dp, global_batching_dp

    if args.hb in ('random', 'metis', 'fennel'):
        node_blocks, assigns = make_blocks(data, train_nid, args.num_blocks, mode=args.hb)
        print(f"edge_cuts = {edge_cuts(data, assigns)}/{data.num_edges}")
        nid_dp = hier_batching_dp(
            node_blocks,
            args.num_blocks//args.block_ratio,
            args.batch_size,
        )
        print(f"{len(node_blocks)} blocks, {len(node_blocks)//args.block_ratio} per batch")
        train_dp = make_ns_dp(data, nid_dp, fanout=args.fanout)
    else:
        nid_dp = global_batching_dp(train_nid, args.batch_size, shuffle=True)
        train_dp = make_ns_dp(data, nid_dp, fanout=args.fanout)
    # elif args.hb == 'once':
    #     train_nid_dp = global_batching_dp(train_nid, args.batch_size, shuffle=False)
    #     train_dp = make_ns_dp(data, train_nid_dp, args.fanout, shuffle=True, batch_size=args.batch_size)

    run(data, train_dp, val_nid, test_nid, args)
    # val_nid_dp = global_batching_dp(val_nid, batch_size=args.batch_size, shuffle=False)
    # val_dp = make_hb_dp(data, val_nid_dp, args.test_fanout)
    # test_nid_dp = global_batching_dp(test_nid, batch_size=args.batch_size, shuffle=False)
    # test_dp = make_hb_dp(data, test_nid_dp, args.test_fanout)

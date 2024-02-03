import sys, os, time, torch
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask, degree, mask_to_index
from torch_sparse import SparseTensor
from data.graphloader import serialize, NodePropPredDataset, ChunkedNodePropPredDataset
import data.partitioner as P
from data.io import TensorMeta, MmapTensor
from data.ops import scatter, scatter_append, index_select, edge_cuts
from graphutils.rw import lazy_rw, edge_importance
import utils

import logging
logger = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d[%(levelname)s] %(module)s: %(message)s",
    datefmt='%0y-%0m-%0d %0H:%0M:%0S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

class FennelDegOrderPartitioner(P.FennelPartitioner):
    def __init__(self, g, psize, name='Fennel-deg', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices

class FennelWLBPartitioner(P.FennelStrataPartitioner):
    def __init__(self, g, psize, train_mask, name='Fennel-strata-deg', **kwargs):
        super().__init__(g, psize, name=name, **kwargs)
        # overwrite node_order: training nodes first, high-degree first
        degrees = self.rowptr[1:] - self.rowptr[:-1]
        self.node_order = torch.sort(degrees, descending=True).indices
        # degrees = self.rowptr[1:] - self.rowptr[:-1]
        # train_nids = mask_to_index(train_mask)
        # train_deg = degrees[train_mask]
        # train_order = train_nids[torch.sort(train_deg, descending=True).indices]
        # nontrain_nids = mask_to_index(~train_mask)
        # nontrain_deg = degrees[~train_mask]
        # nontrain_order = nontrain_nids[torch.sort(nontrain_deg, descending=True).indices]
        # self.node_order = torch.cat([train_order, nontrain_order])

def get_partitioner(dataset: NodePropPredDataset, args):
    data = dataset[0]
    train_nid = dataset.get_idx_split('train')
    train_mask = index_to_mask(train_nid, size=data.size(0))
    if args.part == 'rand':
        return P.RandomNodePartitioner(data, args.pn)
    elif args.part == 'metis':
        return P.MetisNodePartitioner(data, args.pn)
    elif args.part == 'metis-tb':
        # only balance training nodes in each partition, thus the partition sizes vary a lot
        return P.MetisWeightedPartitioner(data, args.pn, train_nid)
    elif args.part == 'metis-w':
        e_w = edge_importance(data, train_nid, k=args.k)
        return P.MetisWeightedPartitioner(data, args.pn, edge_weights=e_w)
    elif args.part == 'metis-wtb':
        e_w = edge_importance(data, train_nid, k=args.k)
        return P.MetisWeightedPartitioner(data, args.pn, node_weights=train_nid, edge_weights=e_w)
    else:
        pred_labels = data.y.flatten().clone().int()
        num_labels = pred_labels.max().item() + 1
        pred_labels[~train_mask] = num_labels
        if args.part == 'fennel':
            # vanilla version of fennel
            return P.ReFennelPartitioner(
                data, args.pn, slack=1.1, runs=3,
                base=FennelDegOrderPartitioner,
            )
        elif args.part == 'fennel-w':
            # weighted version of fennel
            e_w = edge_importance(data, train_nid, k=args.k)
            return P.ReFennelPartitioner(
                data, args.pn, weights=e_w, slack=1.1, runs=3,
                base=FennelDegOrderPartitioner,
            )
        elif args.part == 'fennel-lb':
            # fennel-lb balances all labels in the training set and non-training nodes
            return P.ReFennelPartitioner(
                data, args.pn, slack=1.1, runs=3,
                base=FennelWLBPartitioner,
                train_mask=train_mask,
                stratify_labels=pred_labels,
                balance_labels=train_mask.int(),
            )
        elif args.part == 'fennel-wlb':
            e_w = edge_importance(data, train_nid, k=args.k)
            return P.ReFennelPartitioner(
                data, args.pn, weights=e_w, slack=1.1, runs=3,
                base=FennelWLBPartitioner,
                train_mask=train_mask,
                stratify_labels=pred_labels,
                balance_labels=train_mask.int(),
            )
        else:
            raise ValueError(args.part)

def get_partition_dir(root, args):
    return os.path.join(root, f"{args.part}-P{args.pn}")

def get_pivots_dir(root, args):
    return os.path.join(root, f"{args.part}-P{args.pn}-pivots")

def partition_dataset(data, n_assigns, args, dataset_dir):
    '''
    partition the dataset edge_index & features based on the partition assignment `n_assigns`
    node ids are relabeled in a way that within each partition, the new node id's are contiguous, e.g.
    old node id:
    part_0: [n_0, n_1, ..., n_(k-1)], part_1: [n_k, n_(k+1), ...], ...
    new node id:
    part_0: [0, 1, ..., k-1], part_1: [k, k+1, ...]
    '''
    partition_dir = get_partition_dir(dataset_dir, args)
    os.makedirs(partition_dir, exist_ok=True)
    N_P = args.pn
    E_P = N_P * N_P

    # scatter nodes
    logger.info("Scatter node data...")
    nids = torch.arange(data.num_nodes) # original node id
    new_x = MmapTensor(TensorMeta.like(data.x, path=partition_dir).random_())
    new_x, node_interval, relabel_nids = scatter_append(
        dim=0, index=n_assigns, src=data.x, max_bin=N_P, out=new_x
    )
    new_y = scatter(index=relabel_nids, src=data.y)
    node_map = scatter(index=relabel_nids, src=nids)

    # 2D partitioning of edges based on n_assigns
    edge_src, edge_dst = data.edge_index
    buf_meta = TensorMeta.like(edge_src, dtype=torch.int32).temp_().random_()
    buf = MmapTensor(buf_meta.clone()), MmapTensor(buf_meta.clone())
    dst_assigns = index_select(n_assigns, index=edge_dst, out=buf[0])
    src_assigns = index_select(n_assigns, index=edge_src, out=buf[1])
    src_assigns *= args.pn
    src_assigns += dst_assigns
    e_assigns = src_assigns

    # scatter edges
    logger.info("Scatter edge data...")
    new_src = MmapTensor(TensorMeta.like(edge_src, path=partition_dir).random_())
    new_dst = MmapTensor(TensorMeta.like(edge_dst, path=partition_dir).random_())
    new_edge_attr = None
    _, edge_interval, scatter_index = scatter_append(
        dim=0, index=e_assigns, src=edge_src, max_bin=E_P, out=new_src
    )
    scatter(index=scatter_index, src=edge_dst, out=new_dst)
    if getattr(data, 'edge_attr', None) is not None:
        new_edge_attr = MmapTensor(TensorMeta.like(data.edge_attr, path=partition_dir).random_())
        scatter(index=scatter_index, src=data.edge_attr, out=new_edge_attr)
    # relabel edges
    # TODO: we could use CSC formats to compress the diagonal partitions to save space & IO
    # but for non-diagonal partition, COO is more likely to be efficient
    index_select(src=relabel_nids, index=new_src, out=new_src)
    index_select(src=relabel_nids, index=new_dst, out=new_dst)

    # write to disk
    data_dict = {
        "num_nodes": data.num_nodes,
        "node_parts": node_interval,
        "node_feat": new_x,
        "labels": new_y,
        "node_map": node_map,       # map: new id to old id
        "node_assign": n_assigns,   # partition id of nodes (ordered by old ids)
        "graph": [{
            "format": "coo",
            "edge_index": [new_src, new_dst, edge_interval],
            "edge_feat": new_edge_attr,
            # "edge_parts_layout": "2D.SRC_MAJOR",
        }]
    }
    dataset.attr_dict['N_P'] = N_P
    dataset.attr_dict['E_P'] = E_P
    processed_dict = {
        "attr": dataset.attr_dict,
        "data": data_dict,
        "idx": dataset.get_idx_split(),
    }
    return partition_dir, serialize(processed_dict, partition_dir)

def get_pivots(dataset: NodePropPredDataset, args):
    targets = dataset.get_idx_split()['train']
    data = dataset[0]
    adj_t = data.adj_t
    if args.pivot == 'topk':
        # NOTE: we should really use adj not adj_t, but since the graph is symmetric...
        # adj_t = SparseTensor(row=dst, col=src, is_sorted=True, sparse_sizes=data.size())
        init_score = torch.zeros(data.size(0))
        init_score[targets] = 1.0
        score = lazy_rw(adj_t, init_score, k=args.k, alpha=0.5)
        pivots = score.topk(int(data.size(0) * args.topk)).indices
    else:
        pivots = targets
    inter_adj = adj_t[pivots]
    intra_adj = inter_adj[:,pivots]
    logger.info("pivots(nodes/inter/intra): {}, {}, {}".format(
        pivots.size(0), inter_adj.nnz(), intra_adj.nnz()))
    return pivots, data.x[pivots], inter_adj, intra_adj

def partition_pivots(chunk_dataset, pivot_data, args, dataset_dir):
    pivot_dir = get_pivots_dir(dataset_dir, args)
    os.makedirs(pivot_dir, exist_ok=True)

    n_assigns = chunk_dataset.node_assign
    pivots, pivots_x, inter_adj, intra_adj = pivot_data
    inter_dst, inter_src, _ = inter_adj.coo()
    intra_dst, intra_src, _ = intra_adj.coo()
    pivots_assign = n_assigns[pivots]

    # 1D partition inter_adj based on src
    inter_src, inter_interval, scatter_index = scatter_append(
        dim=0, index=n_assigns[inter_src], src=inter_src, max_bin=args.pn)
    inter_dst = scatter(index=scatter_index, src=inter_dst)
    # relabel new_src
    relabel = torch.empty_like(chunk_dataset.node_map)
    relabel[chunk_dataset.node_map] = torch.arange(0, chunk_dataset.num_nodes)
    inter_src = relabel[inter_src]
    logger.debug("pivot inter_adj partitions:\n{}".format(inter_interval))
    # 1D partition intra_adj based on src
    intra_src, intra_interval, scatter_index = scatter_append(
        dim=0, index=pivots_assign[intra_src], src=intra_src, max_bin=args.pn
    )
    intra_dst = scatter(index=scatter_index, src=intra_dst)
    logger.debug("pivot intra_adj partitions:\n{}".format(intra_interval))

    # write to disk
    data_dict = {
        "num_nodes": pivots.size(0),
        "node_feat": pivots_x,
        "node_assign": pivots_assign,
        "node_map": pivots,
        "graph": [{
            "format": "coo",
            "label": "edge_index_inter",
            "edge_index": [inter_src, inter_dst, inter_interval],
            # "edge_parts_layout": "1D.SRC",
        }, {
            "format": "coo",
            "label": "edge_index_intra",
            "edge_index": [intra_src, intra_dst, intra_interval],
            # "edge_parts_layout": "1D.SRC",
        }],
    }
    processed_dict = {
        "attr": {
            'N_P': args.pn,
            'E_P': args.pn,
        },
        "data": data_dict,
    }
    return pivot_dir, serialize(processed_dict, pivot_dir)


if __name__ == "__main__":
    import argparse, tqdm
    parser = argparse.ArgumentParser(
        description="transform datasets into formats accepted by NodePropPredDataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, # required=True,
                        default='ogbn-papers100M', help="dataset name")
    parser.add_argument('--root', type=str, default=os.environ.get('DATASETS', None),
                        help="location of transformed datasets")
    parser.add_argument('--part', type=str, default='fennel-wlb',
                        help="graph partitioner")
    parser.add_argument('--pn', type=int, default=1024,
                        help="the partition number")
    parser.add_argument('--pivot-only', action='store_true')
    parser.add_argument('--k', type=int, default=3,
                        help="lazy rw steps")
    parser.add_argument('--pivot', type=str, default='topk',
                        help="topk or train")
    parser.add_argument('--topk', type=float, default=0.01,
                        help="ratio of pivotal nodes")
    parser.add_argument('--check', action="store_true",
                        help="Check the validity of partititioned dataset")
    args = parser.parse_args()
    logger.info(args)

    dataset_dir = os.path.join(args.root, args.dataset.replace('-', '_'))
    dataset = NodePropPredDataset(dataset_dir, mmap=(True,True), formats=('coo', 'csc'))
    data = dataset[0]
    #  data.adj_t = NodePropPredDataset(dataset_dir, mmap=True, random=True, formats=('csc'))[0].adj_t

    if not args.check and not args.pivot_only:
        logger.info(f"#nodes: {data.num_nodes}, #edges: {data.adj_t.nnz()}")

        logger.info(f"Partition graph...")
        tic = time.time()
        n_assigns = get_partitioner(dataset, args).partition()
        toc = time.time()
        # torch.save(n_assigns, "partition.pt")
        logger.info(f"Graph partitoning done, takes {toc-tic:.2f}s")
        logger.info(f"Partition #cuts={edge_cuts(data.edge_index, n_assigns)}")
        _, node_interval, _ = scatter_append(dim=0, index=n_assigns, src=n_assigns, max_bin=args.pn)
        sizes = node_interval[1:] - node_interval[:-1]
        logger.info(f"Partition sizes: avg={int(sizes.float().mean())}, "
              f"min={sizes.min().item()}, max={sizes.max().item()}")

        logger.info("Partition dataset...")
        with utils.parallelism(4):
            tic = time.time()
            partition_dataset(data, n_assigns, args, dataset_dir)
            toc = time.time()
        logger.info(f"Dataset partitioning done, takes {toc-tic:.2f}s")

    if not args.check:
        if args.pivot == 'topk' and args.topk == 0: pass
        else:
            # select pivotal nodes, edges, and partition them
            partition_dir = get_partition_dir(dataset_dir, args)
            logger.info("Select pivots...")
            tic = time.time()
            pivot_data = get_pivots(dataset, args)
            chunked = ChunkedNodePropPredDataset(partition_dir)
            pivots_dir, _ = partition_pivots(chunked, pivot_data, args, dataset_dir)
            toc = time.time()
            logger.info(f"Selection done, takes {toc-tic:.2f}s")

    if args.check:
        def check_ndata(ndata_orig: torch.Tensor, ndata_shfl: torch.Tensor, parts):
            logger.info("Checking ndata")
            with utils.parallelism(4):
                offset = 0
                for i in tqdm.tqdm(range(len(parts))):
                    p_size = len(parts[i])
                    p_nodes = parts[i]
                    data_orig = ndata_orig[p_nodes]
                    data_orig[data_orig.isnan()] = -1
                    data_shfl = ndata_shfl[offset:offset+p_size]
                    data_shfl[data_shfl.isnan()] = -1
                    assert (data_orig == data_shfl).all(), \
                        f"Partition {i}"
                    offset += p_size
            logger.info("Passed")

        def intervalize(offsets: torch.Tensor):
            return torch.vstack([offsets[:-1], offsets[1:]]).t()

        class PartitionSequence:
            def __init__(self, data, intervals):
                self.data = data
                self.intervals = intervals

            def __len__(self):
                return self.intervals.size(0)

            def __getitem__(self, i):
                start, end = self.intervals[i]
                return self.data[start:end]

        partition_dir = get_partition_dir(dataset_dir, args)
        chunked = ChunkedNodePropPredDataset(partition_dir)
        chunked_data = chunked[0]
        n_assigns = chunked.node_assign[chunked.node_map]
        assert (n_assigns[1:] >= n_assigns[:-1]).all()
        node_parts = intervalize(chunked.node_parts)
        node_partitions = PartitionSequence(chunked.node_map, node_parts)
        src_partitions = PartitionSequence(chunked_data.edge_index[0], intervalize(chunked_data.edge_index[-1]))

        train_nodes = chunked.get_idx_split('train')
        train_mask = index_to_mask(train_nodes, chunked.num_nodes)
        remapped_partitions = PartitionSequence(torch.arange(chunked.num_nodes), node_parts)
        train_sizes = [int(train_mask[part].sum()) for part in remapped_partitions]
        logger.info(f"Training nodes per partition: \n{train_sizes}")
        logger.info(f"Partition #cuts={edge_cuts(chunked_data.edge_index, n_assigns)}")

        logger.info("Checking node feat")
        check_ndata(data.x, chunked_data.x, node_partitions)
        check_ndata(data.y, chunked_data.y, node_partitions)
        logger.info("Checking src")
        for i, part in enumerate(tqdm.tqdm(src_partitions)):
            assert (n_assigns[part] == (i // args.pn)).all()
        logger.info("Passed")

        dst_partitions = PartitionSequence(chunked_data.edge_index[1], intervalize(chunked_data.edge_index[-1]))
        logger.info("Checking dst")
        for i, part in enumerate(tqdm.tqdm(dst_partitions)):
            assert (n_assigns[part] == (i % args.pn)).all()
        logger.info("Passed")

        pivots_dir = get_pivots_dir(dataset_dir, args)
        pivot = ChunkedNodePropPredDataset(pivots_dir)
        pivot_data = pivot[0]
        pivot_partitions = PartitionSequence(
            pivot_data.edge_index_inter[0], intervalize(pivot_data.edge_index_inter[-1]))
        logger.info("Checking pivot src")
        for i, part in enumerate(tqdm.tqdm(pivot_partitions)):
            if i != args.pn:
                assert (n_assigns[part] == i).all(), f"Partition {i}"
        logger.info("Checking pivot dst")
        global_degree = degree(data.edge_index[1], data.num_nodes)
        pivot_degree = degree(pivot_data.edge_index_inter[1], pivot_data.num_nodes)
        assert (global_degree[pivot.node_map] == pivot_degree).all()
        logger.info("Passed")

    chunked.drop_mmaps()
    dataset.drop_mmaps()

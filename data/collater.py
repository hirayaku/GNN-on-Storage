from typing import Union, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.utils import mask_to_index
from torch_sparse import SparseTensor
from utils import sort, parallelism
from data.graphloader import ChunkedNodePropPredDataset
from data.ops import scatter_append, ranges_gather, ranges_add, coo_ranges_merge
import logging
logger = logging.getLogger()

def get_edge_parts(parts: torch.Tensor, P:int):
    edge_parts = torch.outer(parts, torch.ones(parts.size(0)).long() * P)
    return edge_parts + parts

def intervalize(offsets: torch.Tensor):
    return torch.vstack([offsets[:-1], offsets[1:]])

class Collator:
    '''
    merge partitions from the chunked graph dataset into a single graph dataset
    * `split`: which data splits to treat as targets
    * `merge_nid`: whether to consolidate node IDs.
    If merge_nid is True, the data will be explicitly loaded into the memory
    Otherwise, we use storage-backed tensor (mmap/direct-io) in graph datasets
    '''
    def __init__(
        self,
        chunked: ChunkedNodePropPredDataset,
        split: Union[str, list[str], tuple[str]]='train',
        merge_nid: bool = True,
    ):
        self.data = chunked[0]
        self.node_offsets = intervalize(chunked.node_parts)
        self.edge_offsets = intervalize(self.data.edge_index[-1])
        self.merge_nid = merge_nid
        self.P = chunked.N_P
        self.idx_mask = torch.zeros(chunked.num_nodes, dtype=torch.bool)
        if not isinstance(split, list) and not isinstance(split, tuple):
            split = (split,)
        for s in split:
            split_idx = chunked.get_idx_split(s)
            self.idx_mask[split_idx] = True

    def batch_nodes(self, batch: torch.Tensor) -> torch.Tensor:
        node_intervals = self.node_offsets[:, batch]
        nodes = [torch.arange(start, end) for start, end in node_intervals.t()]
        return torch.cat(nodes)

    def collate(self, batch: torch.Tensor):
        '''
        Collate selected node partitions and edge partitions into a macro-batch
        * `batch`: the indices of node partitions to collate
        '''
        B = batch.size(0)
        logger.debug(f"Collate {B} parts")
        # construct macro-batch edge index
        batch_nodes = self.batch_nodes(batch)
        src, dst = self.data.edge_index[:2]
        node_intervals = self.node_offsets[:, batch]
        node_part_sizes = node_intervals[1] - node_intervals[0]
        edge_batch = get_edge_parts(batch, self.P).flatten()
        edge_intervals = self.edge_offsets[:, edge_batch]
        edge_part_sizes = edge_intervals[1] - edge_intervals[0]

        logger.debug(f"Constructing macro-batch")
        try:
            if self.merge_nid:
                # relabel nodes: nodes in the interval batch are relabeled starting from 0 in order
                old_nids = node_intervals[0]
                new_nids = node_part_sizes.cumsum(0) - node_part_sizes
                edge_offsets = edge_part_sizes.cumsum(0) - edge_part_sizes
                gathered_src = ranges_gather(src, edge_intervals[0], edge_part_sizes)
                gathered_dst = ranges_gather(dst, edge_intervals[0], edge_part_sizes)
                offset_nids = (new_nids - old_nids).repeat_interleave(B)
                gathered_src = ranges_add(gathered_src, edge_offsets, edge_part_sizes, offset_nids)
                offset_nids = (new_nids - old_nids).repeat(B)
                gathered_dst = ranges_add(gathered_dst, edge_offsets, edge_part_sizes, offset_nids)
                logger.debug(f"Partition Adj gathered")
                num_nodes = node_part_sizes.sum().item()
                edge_index, colptr, _ = coo_ranges_merge(
                    num_nodes, (gathered_src, gathered_dst), edge_offsets, edge_part_sizes)
                row, _ = edge_index
                adj_t = SparseTensor(rowptr=colptr, col=row, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
                targets = mask_to_index(
                    ranges_gather(self.idx_mask, node_intervals[0], node_part_sizes))
                logger.debug(f"Edge-index constructed: {batch_nodes.size(0)} nodes, {row.size(0)} edges, {targets.size(0)} train")

                batch_x = ranges_gather(self.data.x, node_intervals[0], node_part_sizes)
                batch_y = ranges_gather(self.data.y, node_intervals[0], node_part_sizes)
                logger.debug(f"Features gathering done")
                subgraph = Data(
                    num_nodes=num_nodes,
                    part_sizes=node_part_sizes,
                    x=batch_x, y=batch_y,
                    # edge_index=edge_index,
                    adj_t=adj_t,
                )
            else:
                # don't relabel nodes, thus get original node IDs for train nodes
                edge_index, colptr, _ = coo_ranges_merge(
                    self.data.num_nodes, (src, dst), edge_intervals[0], edge_part_sizes
                )
                row, _ = edge_index
                adj_t = SparseTensor(rowptr=colptr, col=row, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
                targets = torch.zeros_like(self.idx_mask)
                targets[batch_nodes] = self.idx_mask[batch_nodes]
                targets = mask_to_index(targets)
                logger.debug(f"Edge-index constructed: {batch_nodes.size(0)} nodes, {row.size(0)} edges, {targets.size(0)} train")
                subgraph = Data(
                    num_nodes=self.data.num_nodes,
                    x=self.data.x, y=self.data.y,
                    # edge_index=edge_index,
                    adj_t=adj_t,
                )
        finally:
            logger.debug(f"Construction done")
            return subgraph, targets

class CollatorPivots(Collator):
    def __init__(
        self,
        chunked: ChunkedNodePropPredDataset,
        pivots: ChunkedNodePropPredDataset,
        split: Union[str, list[str], tuple[str]]='train',
    ):
        super().__init__(chunked, split, merge_nid=True)
        # load pivots dataset
        self.data_pvt = pivots[0]
        self.node_parts_pvt, node_offsets_pvt, _ = scatter_append(dim=0,
            index=pivots.node_assign, src=torch.arange(0, pivots.num_nodes), max_bin=pivots.N_P)
        self.node_offsets_pvt = intervalize(node_offsets_pvt)
        self.edge_offsets_inter = intervalize(self.data_pvt.edge_index_inter[-1])
        self.edge_offsets_intra = intervalize(self.data_pvt.edge_index_intra[-1])

        self.pivot_buf = torch.zeros(self.data_pvt.num_nodes, dtype=torch.int)
        self.pivot_mask = torch.ones(self.data_pvt.num_nodes, dtype=torch.bool)

    def exclude_intervals(self, exclude: torch.Tensor):
        assert (exclude < self.P).all()
        to_exclude = sort(exclude)[0].tolist()
        starts, ends = [], []
        i = 0
        for j in to_exclude:
            if j > i:
                # emit [i, j-1]
                starts.append(i)
                ends.append(j-1)
            i = j + 1
        if i < self.P:
            starts.append(i)
            ends.append(self.P-1)
        return starts, ends

    def batch_pivots(self, batch: torch.Tensor) -> torch.Tensor:
        n_intervals_pvt = self.node_offsets_pvt[:, batch]
        batch_count = 0
        for start, end in n_intervals_pvt.t().tolist():
            size = end - start
            self.pivot_buf[batch_count:batch_count+size] = self.node_parts_pvt[start:end]
            batch_count += size
        return self.pivot_buf[:batch_count]

    def collate(self, batch:torch.Tensor):
        '''
        Collate selected node partitions and edge partitions into a macro-batch
        * `batch`: the indices of node partitions to collate
        '''
        B = batch.size(0)
        # XXX debugging purposes
        batch = batch.sort().values
        logger.debug(f"Collate {B} parts: {batch}")

        # construct macro-batch edge index
        batch_nodes = self.batch_nodes(batch)
        num_main_nodes = batch_nodes.size(0)
        num_pvt_nodes = self.data_pvt.num_nodes
        num_nodes = num_main_nodes + num_pvt_nodes
        # the main edge_index
        src, dst = self.data.edge_index[:2]
        n_intervals = self.node_offsets[:, batch]
        n_part_sizes = n_intervals[1] - n_intervals[0]
        e_batch = get_edge_parts(batch, self.P).flatten()
        e_intervals = self.edge_offsets[:, e_batch]
        e_part_sizes = e_intervals[1] - e_intervals[0]
        # the pivot edge_index_inter
        inter_src, inter_dst = self.data_pvt.edge_index_inter[:2]
        e_intervals_inter = self.edge_offsets_inter[:, batch]
        e_part_sizes_inter = e_intervals_inter[1] - e_intervals_inter[0]
        # the pivot edge_index_intra
        intra_src, intra_dst = self.data_pvt.edge_index_intra[:2]
        remains = self.exclude_intervals(batch)
        e_starts_intra = self.edge_offsets_intra[0][remains[0]]
        e_stops_intra = self.edge_offsets_intra[1][remains[1]]
        e_part_sizes_intra = e_stops_intra - e_starts_intra
        # NOTE: use filtering for now
        to_remove = self.batch_pivots(batch)
        self.pivot_mask[:] = True
        self.pivot_mask[to_remove] = False
        logger.debug(f"Redundant pivots: {to_remove.size(0)}")

        # the main edge_index
        old_nids = n_intervals[0]
        new_nids = n_part_sizes.cumsum(0) - n_part_sizes
        edge_offsets = e_part_sizes.cumsum(0) - e_part_sizes
        gathered_src = ranges_gather(src, e_intervals[0], e_part_sizes)
        gathered_dst = ranges_gather(dst, e_intervals[0], e_part_sizes)
        offset_nids = (new_nids - old_nids).repeat_interleave(B)
        gathered_src = ranges_add(gathered_src, edge_offsets, e_part_sizes, offset_nids)
        offset_nids = (new_nids - old_nids).repeat(B)
        gathered_dst = ranges_add(gathered_dst, edge_offsets, e_part_sizes, offset_nids)
        targets = mask_to_index(
            ranges_gather(self.idx_mask, n_intervals[0], n_part_sizes))
        logger.debug(f"Partition Adj: n={num_main_nodes}, m={gathered_src.size(0)}")
        #  logger.debug(f"Num_main: {num_main_nodes}, max row/col: {gathered_src.max()}, {gathered_dst.max()}")
        # the pivot edge_index_inter
        e_offsets_inter = e_part_sizes_inter.cumsum(0) - e_part_sizes_inter
        inter_src = ranges_gather(inter_src, e_intervals_inter[0], e_part_sizes_inter)
        inter_dst = ranges_gather(inter_dst, e_intervals_inter[0], e_part_sizes_inter)
        inter_src = ranges_add(inter_src, e_offsets_inter, e_part_sizes_inter, new_nids-old_nids)
        inter_dst, indices = sort(inter_dst)
        edge_mask = self.pivot_mask[inter_dst]
        inter_dst = inter_dst[edge_mask]
        inter_dst += num_main_nodes
        inter_src = inter_src[indices]
        inter_src = inter_src[edge_mask]
        inter_size = inter_src.size(0)
        logger.debug(f"Inter Adj constructed: m={inter_size}")
        inter_dst_t, indices = sort(inter_src)
        inter_src_t = inter_dst[indices]
        logger.debug(f"Inter Adj transposed")
        # the pivot edge_index_intra
        intra_src = ranges_gather(intra_src, e_starts_intra, e_part_sizes_intra)
        intra_src += num_main_nodes
        intra_dst = ranges_gather(intra_dst, e_starts_intra, e_part_sizes_intra)
        intra_dst += num_main_nodes
        intra_offsets = e_part_sizes_intra.cumsum(0) - e_part_sizes_intra
        logger.debug(f"Intra Adj constructed: m={intra_src.size(0)}")

        edge_index, ptr, _ = coo_ranges_merge(num_nodes,
            [ (inter_src, inter_dst), (inter_src_t, inter_dst_t), (intra_src, intra_dst), 
              (gathered_src, gathered_dst)],
            [torch.tensor([0]), torch.tensor([0]), intra_offsets, edge_offsets],
            [torch.tensor([inter_size]), torch.tensor([inter_size]), e_part_sizes_intra, e_part_sizes],
        )
        row, _ = edge_index
        #  logger.debug(f"Num_nodes: {num_nodes}, max ptr/row: {ptr.size(0)-1}, {row.max()}")
        adj_t = SparseTensor(rowptr=ptr, col=row, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
        logger.debug(f"Macro-batch graph constructed: n={num_nodes}, m={row.size(0)}")
        del intra_src, intra_dst, inter_src, inter_dst, inter_src_t, inter_dst_t
        del gathered_src, gathered_dst

        batch_x_shape = list(self.data.x.shape)
        batch_x_shape[0] = num_nodes
        batch_x = torch.empty(batch_x_shape, dtype=self.data.x.dtype)
        batch_x = ranges_gather(self.data.x, n_intervals[0], n_part_sizes, out=batch_x)
        batch_x[num_main_nodes:] = self.data_pvt.x
        batch_y_shape = list(self.data.y.shape)
        batch_y_shape[0] = num_nodes
        batch_y = torch.zeros(batch_y_shape, dtype=self.data.y.dtype)
        batch_y = ranges_gather(self.data.y, n_intervals[0], n_part_sizes, out=batch_y)
        logger.debug(f"Features gathering done")
        return Data(
            num_nodes=num_nodes,
            num_pivots=num_pvt_nodes,
            part_sizes=n_part_sizes,
            x=batch_x, y=batch_y,
            # edge_index=edge_index,
            adj_t=adj_t,
        ), targets

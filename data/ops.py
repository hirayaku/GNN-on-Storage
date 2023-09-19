import torch
from typing import Optional, Tuple, Union
from data.io import TensorMeta, Dtype, MmapTensor

def scatter(
        index: torch.Tensor,
        src: torch.Tensor,
        out: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(src)
    return torch.ops.xTensor.scatter_copy(out, index, src)

def scatter_append(
        dim: int,
        index: torch.Tensor,
        src: torch.Tensor,
        max_bin: Optional[int]=None,
        out: Optional[torch.Tensor]=None,
    ):
    '''
    mimics torch.scatter, except that the reduction operator is the non-commutative "append"
    returns an offset tensor containing the start offset of each group in the output tensor
    - out: the scattered result from src
    - hist_cum: the start offsets of each group
    - scatter_pos: the output index of each item in src
    '''
    assert src.size(0) == index.size(0), f"size mismatch: {src.size()}, {index.size()}"
    max_bin = int(index.max()) + 1 if max_bin is None else max_bin
    if out is None:
        out = torch.empty_like(src)
    hist = index.bincount(minlength=max_bin)
    hist_cum = torch.zeros(max_bin+1, dtype=torch.long)
    torch.cumsum(hist, dim=0, out=hist_cum[1:])
    scatter_pos = MmapTensor(TensorMeta.like(index, dtype=torch.long).temp_())
    torch.ops.xTensor.scatter_index(scatter_pos, index, hist_cum)
    scatter(scatter_pos, src, out)
    return out, hist_cum, scatter_pos

def ranges_gather(
        src: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        out: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
    '''
    Gather slices src[starts[i] : starts[i] + lengths[i]] for i = 0, ..., N-1
    and store the results into `out`
    '''
    out_size = int(lengths.sum())
    assert starts.size(0) == lengths.size(0)
    assert (starts < src.size(0)).all(), "out of range"
    assert (lengths >= 0).all(), "invalid length"
    if out is None:
        out_shape = list(src.shape)
        out_shape[0] = out_size
        out = torch.empty(out_shape, dtype=src.dtype, device=src.device)
    assert out.size(0) >= out_size, "Tensor `out` can't fit"
    return torch.ops.xTensor.ranges_gather(out, src, starts, lengths)

def ranges_add(
        targets: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
    '''
    In-place addition to the `targets` tensor
    '''
    assert starts.size(0) == lengths.size(0)
    assert starts.size(0) == values.size(0)
    assert (starts < targets.size(0)).all(), "out of range"
    assert (lengths >= 0).all(), "invalid length"
    assert (starts + lengths <= targets.size(0)).all(), "out of range"
    return torch.ops.xTensor.ranges_add(targets, starts, lengths, values)

def index_select(
        src: torch.Tensor,
        index: torch.Tensor,
        out: Optional[torch.Tensor]=None,
        buf_size=1024**3//8
    ) -> torch.Tensor:
    '''
    index_select for very large `index` and `out`
    '''
    if out is None:
        out = torch.empty_like(src)

    steps = index.size(0)
    row_size = src.numel() // src.size(0)
    buf_size = int(buf_size)
    if buf_size < row_size:
        raise ValueError("buffer is smaller than a row in src!")
    per_blk = buf_size // row_size
    start = 0
    while start + per_blk < steps:
        index_buf = index[start:start+per_blk]
        out[start:start+per_blk] = src[index_buf]
        start += per_blk
    index_buf = index[start:steps]
    out[start:steps] = src[index_buf]
    return out

def coo_list_merge(
        num_nodes: int,
        coo_tensors: list[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Merge coo fragments into a unifying coo.
    NOTE: node IDs in the input coo tensors should be in the range of [0, num_nodes).
    Otherwise undefined behavior will happen (e.g. segfaults)
    '''
    return torch.ops.xTensor.coo_list_merge(num_nodes, coo_tensors)

def coo_ranges_merge(
        num_nodes: int,
        coo_tensors: list[Tuple[torch.Tensor, torch.Tensor]],
        starts: list[torch.Tensor],
        lengths: list[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Merge coo fragments into a unifying coo without explictly constructing a list of coo's.
    NOTE: node IDs in the specified input coo tensors should be in the range of [0, num_nodes).
    Otherwise undefined behavior will happen (e.g. segfaults)
    '''
    if not isinstance(coo_tensors, list):
        assert not isinstance(starts, list)
        assert not isinstance(lengths, list)
        coo_tensors = [coo_tensors]
        starts = [starts]
        lengths = [lengths]
    assert len(coo_tensors) == len(starts)
    assert len(starts) == len(lengths)
    return torch.ops.xTensor.coo_ranges_merge(num_nodes, coo_tensors, starts, lengths)

def edge_cuts(edge_index, n_assigns):
    '''
    Given a partitioning of nodes `n_assigns`, compute the induced edge cuts
    '''
    tinfo = TensorMeta.like(edge_index[0], dtype=n_assigns.dtype).temp_()
    src_assigns = MmapTensor(tinfo.clone())
    index_select(n_assigns, index=edge_index[0], out=src_assigns)
    dst_assigns = MmapTensor(tinfo.clone())
    index_select(n_assigns, index=edge_index[1], out=dst_assigns)
    src_assigns -= dst_assigns
    return (src_assigns != 0).sum().item()

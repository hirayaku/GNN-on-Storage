import torch
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

def degree(sp: SparseTensor) -> torch.Tensor:
    rowptr, _, _ = sp.csr()
    return rowptr[1:] - rowptr[:-1]

def lazy_rw(
    sp: SparseTensor, start: torch.Tensor, deg: OptTensor=None,
    k: int=1, alpha: float=0.5
) -> torch.Tensor:
    if deg is None:
        deg = degree(sp)
    deg = deg.to(sp.device())
    deg[deg==0] = 1
    score = start
    n = sp.size(0)
    for _ in range(k):
        normalized = score / deg
        score = (1-alpha) * score + sp.spmm(normalized.view(n, 1)).view(-1) * alpha
    return score

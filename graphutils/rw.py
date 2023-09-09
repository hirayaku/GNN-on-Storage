import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor

def degree(sp: SparseTensor) -> torch.Tensor:
    rowptr, _, _ = sp.csr()
    return rowptr[1:] - rowptr[:-1]

def lazy_rw(
    adj_t: SparseTensor, start: torch.Tensor, deg: OptTensor=None,
    k: int=1, alpha: float=0.5, return_all=False
) -> torch.Tensor:
    '''
    perform `k`-step lazy random walks from `targets`, compute the probability
    the random walks end at each node. `alpha` is the probability of leaving.
    '''
    assert k >= 0, "requires k >= 0"
    if deg is None:
        deg = degree(adj_t)
    deg = deg.to(adj_t.device())
    deg[deg==0] = 1
    score = start
    score_vec = [score]
    n = adj_t.size(0)
    for _ in range(k):
        normalized = score / deg
        score = (1-alpha) * score + adj_t.spmm(normalized.view(n, 1)).view(-1) * alpha
        score_vec.append(score)
    if return_all:
        return score_vec
    else:
        return score

def node_importance(
    data: Data, targets: torch.Tensor, k: int=1, alpha: float=0.5
) -> torch.Tensor:
    '''
    Perform lazy random walks from `targets`, compute the probability
    a node is included in this random walk.
    Currently, it only works for symmetric graphs.
    '''
    adj_t = data.adj_t
    assert k >= 0, "requires k >= 0"
    n = adj_t.size(0)
    score = torch.zeros((n,))
    score[targets] = 1 / targets.size(0)
    deg = degree(adj_t)
    rw_probs = lazy_rw(adj_t, score, deg=deg, k=k, alpha=alpha, return_all=True)
    if k == 0:
        return score
    elif k == 1:
        return rw_probs[0] + rw_probs[1] - rw_probs[0] * (1-alpha)
    elif k == 2:
        # compute the probability for a vertex to leave and come back to itself after 2 hops
        u, v = data.edge_index
        p = 1 / deg
        p_e = p[u]
        p_e *= p[v]
        adj_new = adj_t.set_value(p_e, 'csr')
        p_revisit = adj_new.spmm(torch.ones((n, 1))).view(-1)
        del p_e, adj_new
        temp = sum(rw_probs)
        temp -= rw_probs[0] * (1-alpha)  # v is visited at hop 0 & 1
        temp -= rw_probs[1] * (1-alpha)  # v is visited at hop 1 & 2
        temp -= rw_probs[0] * (p_revisit * alpha**2) # v at hop 0,2, minus v at hop 0-2
        # temp += rw_probs[0] * (1-alpha)**2
        return temp
    else:
        raise RuntimeError("up to k == 2 for node importance")

def edge_importance(
    data: Data,  targets: torch.Tensor, k:int=1, alpha: float=0.5,
) -> torch.Tensor:
    '''
    compute importance scores for edges in the graph `adj_t` for `targets`.
    Currently, it only works for symmetric graphs.
    '''
    node_impt = node_importance(data, targets, k - 1, alpha)
    i_deg = degree(data.adj_t)
    u, v = data.edge_index
    edge_impt = (node_impt * alpha / i_deg)[u]
    edge_impt += (node_impt * alpha / i_deg)[v]
    return edge_impt

if __name__ == '__main__':
    from data.graphloader import NodePropPredDataset
    import data.partitioner as P
    dataset = NodePropPredDataset('/mnt/md0/hb_datasets/ogbn_arxiv', mmap=False)
    data = dataset[0]
    trains = dataset.get_idx_split('train')
    u, v = data.edge_index
    def edge_cuts(assigns):
        return (assigns[u] - assigns[v] != 0).sum()
    # node_impt = node_importance(data, trains, k=2)
    edge_impt = edge_importance(data, trains, k=3)
    assigns = P.MetisWeightedPartitioner(data, 64).partition()
    print(edge_cuts(assigns))
    assigns = P.MetisWeightedPartitioner(
        data, 64, edge_weights=torch.randint(1, 16, (data.adj.nnz(),))
    ).partition()
    print(edge_cuts(assigns))
    assigns = P.MetisWeightedPartitioner(data, 64, edge_weights=edge_impt).partition()
    print(edge_cuts(assigns))

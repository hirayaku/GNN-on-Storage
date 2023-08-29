import time
from functools import lru_cache
from typing import Optional, Tuple
import torch
from torch_geometric.typing import OptTensor
import utils

def group(ids, assigns, psize) -> list[torch.Tensor]:
    '''
    return ids of each part in a list given the assignments
    '''
    assert ids.size(0) == assigns.size(0)
    _, idx = torch.sort(assigns, stable=True)
    shuffled = ids[idx]

    # compute partition sizes
    group_sizes = torch.histc(assigns.float(), bins=psize, min=0, max=psize).long()
    group_offs = torch.cumsum(group_sizes, dim=0)
    groups = [shuffled[:group_offs[0]]] + \
        [shuffled[group_offs[i-1]:group_offs[i]] for i in range(1, len(group_offs))]
    return groups

class NodePartitioner(object):
    '''
    NodePartitioner generates a partition assignment tensor
    for nodes in a graph via the `partition` method
    '''
    def __init__(self, g, psize, name=None):
        self.g = g
        self.psize: int = psize
        self.name = name

    def __str__(self):
        return f"{self.name}-p{self.psize}"

    def partition(self, **kwargs) -> torch.IntTensor:
        raise NotImplementedError

class RandomNodePartitioner(NodePartitioner):
    def __init__(self, g, psize):
        super().__init__(g, psize, name='rand')

    @lru_cache
    def partition(self, **kwargs) -> torch.IntTensor:
        return torch.randint(self.psize, (self.g.size(0),), dtype=torch.int)

# class SeqNodePartitioner(NodePartitioner):
#     def __init__(self, g, psize):
#         super().__init__(g, psize, name='seq')
#     def partition(self, **kwargs) -> torch.IntTensor:
#         size_per_part = self.g.size(0) // self.psize

from torch_sparse.tensor import SparseTensor
'''
# taken from pytorch_sparse's metis.py
def weight2metis(weight: Tensor) -> Optional[Tensor]:
    sorted_weight = weight.sort()[0]
    diff = sorted_weight[1:] - sorted_weight[:-1]
    if diff.sum() == 0:
        return None
    weight_min, weight_max = sorted_weight[0], sorted_weight[-1]
    srange = weight_max - weight_min
    min_diff = diff.min()
    scale = (min_diff / srange).item()
    tick, arange = scale.as_integer_ratio()
    weight_ratio = (weight - weight_min).div_(srange).mul_(arange).add_(tick)
    return weight_ratio.to(torch.long)

def metis_partition(
    src: SparseTensor,
    num_parts: int,
    recursive: bool = False,
    node_weight: Optional[Tensor] = None,
) -> Tensor:

    assert num_parts >= 1
    if num_parts == 1:
        partptr = torch.tensor([0, src.size(0)], device=src.device())
        perm = torch.arange(src.size(0), device=src.device())
        return src, partptr, perm

    rowptr, col, value = src.csr()
    rowptr, col = rowptr.cpu(), col.cpu()

    if value is not None:
        assert value.numel() == col.numel()
        value = value.view(-1).detach().cpu()
        if value.is_floating_point():
            value = weight2metis(value)

    if node_weight is not None:
        assert node_weight.numel() == rowptr.numel() - 1
        node_weight = node_weight.view(-1).detach().cpu()
        if node_weight.is_floating_point():
            node_weight = weight2metis(node_weight)
        cluster = torch.ops.torch_sparse.partition2(rowptr, col, value,
                                                    node_weight, num_parts,
                                                    recursive)
    else:
        cluster = torch.ops.torch_sparse.partition(rowptr, col, value,
                                                   num_parts, recursive)
    cluster = cluster.to(src.device())
    return cluster
'''

from torch_sparse import SparseTensor
from torch_geometric.utils import index_to_mask 
class MetisNodePartitioner(NodePartitioner):
    def __init__(self, g, psize, name='metis'):
        super().__init__(g, psize, name=name)

    @lru_cache
    def partition(
        self,
        node_weights=None,
        edge_weights=None,
        intermediate_dir=utils.SCRATCH_DIR
    ) -> torch.IntTensor:
        if edge_weights is not None and edge_weights.dim() > 1:
            raise ValueError("METIS doesn't support multi-labels for edges")
        if node_weights is not None and node_weights.dim() > 1:
            raise ValueError("DGL's port of METIS doesn't support multi-labels for nodes")
        adj = None
        if hasattr(self.g, 'adj_t'):
            adj = self.g.adj_t
        elif hasattr(self.g, 'adj'):
            adj = self.g.adj
        else:
            edge_index = self.g.edge_index
            adj = SparseTensor(
                row=edge_index[0], col=edge_index[1], value=edge_weights,
                sparse_sizes=(self.g.num_nodes, self.g.num_nodes),
            )
        if not isinstance(adj, SparseTensor):
            adj = SparseTensor(
                rowptr=adj[0], col=adj[1], value=edge_weights,
                sparse_sizes=(self.g.num_nodes, self.g.num_nodes),
            )
        # with utils.cwd(intermediate_dir):
        #     self.assigns = metis_partition(adj, self.psize, node_weight=node_weights)
        #     return self.assigns
        import dgl
        graph = dgl.graph(adj.coo()[:-1])
        with utils.cwd(intermediate_dir):
            return dgl.metis_partition_assignment(
                graph, self.psize, node_weights,
            ).int()

class MetisWeightedPartitioner(MetisNodePartitioner):
    def __init__(
            self, g, psize,
            node_weights: torch.Tensor=None,
            edge_weights: torch.Tensor=None
    ):
        self.e_w = edge_weights
        if node_weights is not None and not node_weights.is_floating_point():
            if node_weights.size(0) < g.size(0):
                self.node_weights = index_to_mask(node_weights, g.size(0)).long()
            else:
                self.node_weights = node_weights.long()
        else:
            self.node_weights = node_weights
        super().__init__(g, psize, name='metis-weighted')

    def partition(self, **kwargs):
        return super().partition(
            node_weights=self.node_weights,
            edge_weights=self.e_w,
            **kwargs)

from typing import Optional, Union

class FennelPartitioner(NodePartitioner):
    def __init__(
        self, g, psize, order: Optional[torch.Tensor]=None,
        gamma=1.5, slack=1.1, alpha_ratio=None,
        name='Fennel', use_opt=True
    ):
        try:
            torch.ops.load_library('data/build/libfennel.so')
        except:
            print("Fail to load Fennel. Did you build it?")
            raise
        super().__init__(g, psize, name=name)
        self.node_order = order
        self.gamma = gamma
        self.slack = slack

        adj = None
        if hasattr(self.g, 'adj_t'):
            adj = self.g.adj_t
        elif hasattr(self.g, 'adj'):
            adj = self.g.adj
        else:
            edge_index = self.g.edge_index
            adj = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                sparse_sizes=(self.g.num_nodes, self.g.num_nodes),
            )
        if isinstance(adj, SparseTensor):
            self.rowptr, self.col, _ = adj.csr()
        else:
            self.rowptr, self.col = adj
        self.n = g.num_nodes
        self.m = self.col.size(0)
        # default alpha = k**(gamma-1) * m / n**gamma
        self.alpha = self.psize**(self.gamma-1) * self.m / self.n**self.gamma
        self.scale_alpha = alpha_ratio if alpha_ratio is not None else 1
        self.use_opt = use_opt

    def partition(self, init_partition=None) -> torch.IntTensor:
        # 64, 1/4
        # 1024, 1/16
        # 16384, 1/16
        thres = min(max(self.psize-64, 0)/15/1024, 1/16)
        if self.use_opt:
            self.assigns = torch.ops.Fennel.partition_opt(
                self.rowptr, self.col, self.psize, self.node_order,
                self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
                thres,
            )
        else:
            self.assigns = torch.ops.Fennel.partition(
                self.rowptr, self.col, self.psize, self.node_order,
                self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
            )
        return self.assigns

class FennelPartitionerPar(FennelPartitioner):
    def __init__(self, g, psize, name='Fennel-par', **kwargs):
        base_kw = {
            k: kwargs[k] for k in ('gamma', 'slack', 'alpha_ratio', 'use_opt', 'labels')
            if k in kwargs
        }
        super().__init__(g, psize, name=name, **base_kw)

    def partition(self, init_partition=None) -> torch.IntTensor:
        self.assigns = torch.ops.Fennel.partition_parallel(
            self.rowptr, self.col, self.psize, self.node_order,
            self.gamma, self.alpha*self.scale_alpha, self.slack,
            init_partition, 1/100,
        )
        return self.assigns

class ReFennelPartitioner(NodePartitioner):
    '''
    Some notes on how to set the base fennel, alpha_ratio & runs
    - a single run of Fennel(deg-order) is generally better than Fennel(rand),
    - for multiple runs, when beta=1 and alpha_ratio=1, Fennel(deg-order) is
    still generally better than Fennel(rand)
    - for multiple runs, when beta>1, set alpha_ratio properly (less than 1) so
    the beginning alpha is smaller while the terminating alpha is larger than default.
    It leads to better partitions according to the paper.
    In this case, Fennel(rand) could be on par with or even better than Fennel(deg-order)

    Meaning of some args:
    * runs: how many streaming passes ReFennel should run
    * beta: the decay factor of alpha between streaming passes
    * kwargs: used to specify all the remaining arguments you'd pass to FennelParitioner
    '''
    def __init__(self, g, psize, runs, name="reFennel", beta=1.0, **kwargs):
        try:
            torch.ops.load_library('data/build/libfennel.so')
        except:
            print("Fail to load Fennel. Did you build it?")
            raise

        super().__init__(g, psize, name)
        base_kw = {
            k: kwargs[k] for k in ('gamma', 'slack', 'alpha_ratio', 'use_opt', 'labels')
            if k in kwargs
        }
        self.runs = runs
        self.beta = beta
        base_fennel_cls = kwargs.get('base', FennelPartitioner)
        self.base_fennel = base_fennel_cls(g, psize, **base_kw)

    def partition(self) -> torch.IntTensor:
        self.assigns = None
        for r in range(self.runs):
            print(f"ReFennel Run#{r}")
            self.assigns = self.base_fennel.partition(init_partition=self.assigns)
            self.base_fennel.alpha *= self.beta
        return self.assigns

class FennelStrataPartitioner(FennelPartitioner):
    def __init__(self, g, psize, labels:OptTensor=None, name='fennel-strata', **kwargs):
        '''
        if `labels` is None, FennelStratified falls backs to vanilla Fennel
        if a node's label < 0, FennelStatified doesn't balance it.
        alphas[L] = m_L * pow(k, gamma-1) / pow(n_L, gamma), where n_L is #nodes with label L,
        and m_L is #edges starting from nodes labeled L
        TODO: handle multi-label cases
        '''
        super().__init__(g, psize, name=name, **kwargs)
        old_alpha = self.alpha
        if labels is not None:
            self.labels = labels.flatten()
            num_labels = int(labels.max()) + 1
            label_hist = torch.histc(self.labels.float(), bins=num_labels, min=0, max=num_labels)
            degree = self.rowptr[1:]-self.rowptr[:-1]
            m_label = torch.zeros((num_labels,), dtype=torch.long)
            m_label.scatter_add_(0, index=labels.long(), src=degree)
            self.alpha = m_label * self.psize**(self.gamma-1) / label_hist**self.gamma
            #  self.alpha = self.m * self.psize**(self.gamma-1) / label_hist**self.gamma
        else:
            self.labels = labels
            self.alpha = torch.tensor([self.alpha])
        # reset outlier alpha values
        self.alpha[self.alpha.isnan()] = 0
        self.alpha[self.alpha.isinf()] = 0
        self.alpha *= self.scale_alpha
        print(self.scale_alpha, old_alpha, self.alpha)

    def partition(self, init_partition=None) -> torch.IntTensor:
        thres = min(max(self.psize-64, 0)/15/1024, 1/16)
        self.assigns = torch.ops.Fennel.partition_strata_opt(
            self.rowptr, self.col, self.psize, self.labels, self.node_order,
            self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
            thres,
        )
        return self.assigns

class FennelStrataPartitionerPar(FennelStrataPartitioner):
    def __init__(self, g, psize, labels=None, name='fennel-strata-par', **kwargs):
        super().__init__(g, psize, labels, name=name, **kwargs)

    def partition(self, init_partition=None) -> torch.IntTensor:
        thres = min(max(self.psize-64, 0)/15/1024, 1/16)
        self.assigns = torch.ops.Fennel.partition_strata_par(
            self.rowptr, self.col, self.psize, self.labels, self.node_order,
            self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
            thres,
        )
        return self.assigns
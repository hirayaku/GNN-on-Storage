from typing import Optional, Tuple
import torch
from torch_geometric.typing import Tensor, OptTensor
from torch_geometric.utils import index_to_mask
from torch_sparse.tensor import SparseTensor
import utils
import logging
logger = logging.getLogger()

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

    def partition(self, **kwargs) -> torch.IntTensor:
        return torch.randint(self.psize, (self.g.size(0),), dtype=torch.int)

def weight2metis(weight: Tensor) -> Optional[Tensor]:
    weight_min = weight.min()
    weight = weight - weight_min
    if weight.sum() == 0:
        return None
    unit = weight[weight!=0].min()
    weight += unit
    return weight.div_(unit).round_().long()

# taken from pytorch_sparse's metis.py but omit the last step
def metis_partition(
    src: SparseTensor,
    num_parts: int,
    recursive: bool = False,
    weighted: bool = False,
    node_weight: Optional[Tensor] = None,
) -> Tensor:

    assert num_parts >= 1
    if num_parts == 1:
        return torch.ones(src.size(0), dtype=torch.int)

    rowptr, col, value = src.csr()
    rowptr, col = rowptr.cpu(), col.cpu()

    if value is not None and weighted:
        assert value.numel() == col.numel()
        value = value.view(-1).detach().cpu()
        if value.is_floating_point():
            value = weight2metis(value)
    else:
        value = None

    if node_weight is not None:
        assert node_weight.numel() == rowptr.numel() - 1
        node_weight = node_weight.view(-1).detach().cpu()
        if node_weight.is_floating_point():
            node_weight = weight2metis(node_weight)
        cluster = torch.ops.torch_sparse.partition2(
                rowptr, col, value, node_weight, num_parts, recursive) # type: ignore
    else:
        cluster = torch.ops.torch_sparse.partition(
                rowptr, col, value, num_parts, recursive) # type: ignore
    cluster = cluster.to(src.device())
    return cluster

class MetisNodePartitioner(NodePartitioner):
    '''
    When the edges are weighted, this partitioner assumes the weights are symmetrized
    '''
    def __init__(self, g, psize, name='metis'):
        super().__init__(g, psize, name=name)

    def partition(
        self,
        node_weights=None,
        edge_weights=None,
        intermediate_dir=utils.SCRATCH_DIR
    ) -> torch.Tensor:
        if edge_weights is not None and edge_weights.dim() > 1:
            raise ValueError("METIS doesn't support multi-labels for edges")
        if node_weights is not None and node_weights.dim() > 1:
            raise ValueError("This port of METIS doesn't support multi-labels for nodes")
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
        if not isinstance(adj, SparseTensor):
            adj = SparseTensor(
                rowptr=adj[0], col=adj[1],
                sparse_sizes=(self.g.num_nodes, self.g.num_nodes),
                is_sorted=True,
            )
        adj = adj.set_value(edge_weights, 'csr')

        with utils.cwd(intermediate_dir):
            return metis_partition(
                adj, self.psize, weighted=True, node_weight=node_weights
            ).int()
            # import dgl
            # graph = dgl.graph(adj.coo()[:-1])
            # return dgl.metis_partition_assignment(
            #     graph, self.psize, node_weights,
            # ).int()

class MetisWeightedPartitioner(MetisNodePartitioner):
    def __init__(
            self, g, psize,
            node_weights: torch.Tensor=None,
            edge_weights: torch.Tensor=None
    ):
        self.edge_weights = edge_weights
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
            edge_weights=self.edge_weights,
            **kwargs)

class FennelPartitioner(NodePartitioner):
    def __init__(
        self, g, psize, weights: OptTensor = None, order: OptTensor =None,
        gamma=1.5, slack=1.1, alpha_ratio=None,
        name='Fennel', use_opt=True
    ):
        try:
            torch.ops.load_library('data/build/libfennel.so')
        except:
            logger.error("Fail to load Fennel. Did you build it?")
        super().__init__(g, psize, name=name)
        self.weights = weights
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
            self.weights = None
        self.n = g.num_nodes
        self.m = self.col.size(0)
        # normalize weights
        if self.weights is not None:
            weights_sum = self.weights.sum()
            self.weights.div_(weights_sum).mul_(self.m)
        # default alpha = k**(gamma-1) * m / n**gamma
        self.alpha = self.psize**(self.gamma-1) * self.m / self.n**self.gamma
        self.scale_alpha = alpha_ratio if alpha_ratio is not None else 1
        self.use_opt = use_opt

    def partition(self, init_partition=None) -> torch.IntTensor:
        self.assigns = torch.ops.Fennel.partition_weighted(
            self.rowptr, self.col, self.weights, self.psize, self.node_order,
            self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
            1/8,
        )
        # if self.use_opt:
        #     self.assigns = torch.ops.Fennel.partition_opt(
        #        self.rowptr, self.col, self.psize, self.node_order,
        #        self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
        #        thres,
        #     )
        # else:
        #     self.assigns = torch.ops.Fennel.partition(
        #         self.rowptr, self.col, self.psize, self.node_order,
        #         self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
        #     )
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
            logger.error("Fail to load Fennel. Did you build it?")
            raise

        super().__init__(g, psize, name)
        base_kw = {
            k: kwargs[k] for k in (
                'weights', 'labels', 'gamma', 'slack', 'alpha_ratio', 'use_opt', 'node_order'
            ) if k in kwargs
        }
        self.runs = runs
        self.beta = beta
        base_fennel_cls = kwargs.get('base', FennelPartitioner)
        self.base_fennel = base_fennel_cls(g, psize, **base_kw)

    def partition(self) -> torch.IntTensor:
        self.assigns = None
        for r in range(self.runs):
            logger.info(f"ReFennel Run#{r}")
            self.assigns = self.base_fennel.partition(init_partition=self.assigns)
            self.base_fennel.alpha *= self.beta
        return self.assigns

class FennelStrataPartitioner(FennelPartitioner):
    def __init__(
            self, g, psize, weights:OptTensor=None, labels:OptTensor=None,
            name='fennel-strata', **kwargs
        ):
        '''
        if `labels` is None, FennelStratified falls backs to vanilla Fennel
        if a node's label < 0, FennelStatified doesn't balance it.
        alphas[L] = m_L * pow(k, gamma-1) / pow(n_L, gamma), where n_L is #nodes with label L,
        and m_L is #edges starting from nodes labeled L
        TODO: handle multi-label cases
        '''
        super().__init__(g, psize, weights=weights, name=name, **kwargs)
        old_alpha = self.alpha
        if labels is not None:
            # assumption: single-label data
            self.labels = labels.flatten()
            num_labels = int(labels.max()) + 1
            label_hist = torch.histc(self.labels.float(), bins=num_labels, min=0, max=num_labels)
            degree = self.rowptr[1:]-self.rowptr[:-1]
            # vanilla fennel
            # self.alpha = self.m * self.psize**(self.gamma-1) / label_hist**self.gamma
            # labeled fennel
            m_label = torch.zeros((num_labels,), dtype=torch.long)
            m_label.scatter_add_(0, index=labels.long(), src=degree)
            self.alpha = m_label * self.psize**(self.gamma-1) / label_hist**self.gamma
        else:
            self.labels = labels
            self.alpha = torch.tensor([self.alpha])
        # reset outlier alpha values
        self.alpha[self.alpha.isnan()] = 0
        self.alpha[self.alpha.isinf()] = 0
        self.alpha *= self.scale_alpha
        logger.debug(self.scale_alpha, old_alpha, self.alpha)

    def partition(self, init_partition=None) -> torch.IntTensor:
        self.assigns = torch.ops.Fennel.partition_strata_weighted(
            self.rowptr, self.col, self.weights, self.psize, self.labels, self.node_order,
            self.gamma, self.alpha*self.scale_alpha, self.slack, 0, init_partition,
            1/8,
        )
        # self.assigns = torch.ops.Fennel.partition_strata_opt(
        #     self.rowptr, self.col, self.psize, self.labels, self.node_order,
        #     self.gamma, self.alpha*self.scale_alpha, self.slack, 0, init_partition,
        #     1/8,
        # )
        return self.assigns

# class FennelStrataPartitionerPar(FennelStrataPartitioner):
#     def __init__(self, g, psize, labels=None, name='fennel-strata-par', **kwargs):
#         super().__init__(g, psize, labels, name=name, **kwargs)

#     def partition(self, init_partition=None) -> torch.IntTensor:
#         thres = min(max(self.psize-64, 0)/15/1024, 1/16)
#         self.assigns = torch.ops.Fennel.partition_strata_par(
#             self.rowptr, self.col, self.psize, self.labels, self.node_order,
#             self.gamma, self.alpha*self.scale_alpha, self.slack, init_partition,
#             thres,
#         )
#         return self.assigns

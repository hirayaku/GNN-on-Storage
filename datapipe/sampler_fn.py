from typing import Callable, Any
import copy, warnings
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.typing import Tensor, OptTensor
from torch_geometric.loader.utils import filter_data

NodeSampler = Callable[[Data, OptTensor, dict], Any]

def filter_and_pin(data, node, row, col, edge) -> Data:
    out = Data()
    out.num_nodes = node.numel()
    # sparse_sizes = out.size()[::-1]
    # # TODO Currently, we set `is_sorted=False`, see:
    # # https://github.com/pyg-team/pytorch_geometric/issues/4346
    # adj_t = SparseTensor(row=col, col=row, sparse_sizes=sparse_sizes,
    #                      is_sorted=False, trust_data=True)
    # out.adj_t = adj_t.pin_memory()
    edge_index = torch.empty([2, col.numel()], dtype=col.dtype, pin_memory=True)
    edge_index[0] = row
    edge_index[1] = col
    out.edge_index = edge_index
    x_shape = list(data.x.shape)
    x_shape[0] = node.numel()
    x = torch.empty(x_shape, dtype=data.x.dtype, pin_memory=True)
    torch.index_select(data.x, dim=0, index=node, out=x)
    out.put_tensor(x, attr_name='x', index=None)
    y_shape = list(data.y.shape)
    y_shape[0] = node.numel()
    y = torch.empty(y_shape, dtype=data.y.dtype, pin_memory=True)
    torch.index_select(data.y, dim=0, index=node, out=y)
    out.put_tensor(y, attr_name='y', index=None)
    return out

def gather_feature(args, filter_fn=None):
    data, input_id, node, row, col, edge, hop_nodes, hop_edges = args
    if filter_fn is None:
        data = filter_data(data, node, row, col, edge, perm=None)
    else:
        data = filter_fn(data, node, row, col, edge)
    data.batch = None
    data.input_id = input_id.pin_memory()
    data.batch_size = input_id.size(0)
    data.num_sampled_nodes = hop_nodes
    data.num_sampled_edges = hop_edges
    return data

class PygNeighborSampler:
    '''
    Wrapper of the neighbor sampler functions used in PyG
    * `filter_data_fn(data, node, row, col, edge, perm)`: a function to specify how to
      slice node/edge data from the input `data` object
    '''
    def __init__(
        self,
        fanout,
        replace=False,
        directed=True,
        return_eid=False,
        filter_data_fn=None,
        filter_per_worker=True,
        unbatch=False,
        **overwrite_kw
    ):
        self.fanout = fanout
        self.replace = replace
        self.directed = directed
        self.return_eid = return_eid
        self.filter_data = filter_data_fn
        self.filter_per_worker = filter_per_worker
        self.unbatch = unbatch
        self.kw = overwrite_kw
        try:
            import pyg_lib
            self.sample = self._pyg_lib_sample
        except ImportError:
            warnings.warn("pyg-lib not found, using the legacy sampler in pytorch_sparse")
            self.sample = self._torch_sparse_sample

    def __call__(self, *args, **kwargs) -> Data:
        if self.unbatch:
            data, input_id = args[0][0]
        else:
            data, input_id = args[0]
        sample_out = self.sample(data, input_id, **kwargs)
        row, col, node, edge, n_nodes, n_edges = sample_out
        args = (data, input_id, node, row, col, edge, n_nodes, n_edges)
        if self.filter_per_worker:
            # slice node and edge attributes
            # data = self.filter_data(data, node, row, col, edge, perm=None)
            # data.batch = None
            # data.input_id = nodes
            # data.batch_size = nodes.size(0)
            # data.num_sampled_nodes = n_nodes
            # data.num_sampled_edges = n_edges
            # return data
            return gather_feature(args, self.filter_data)
        else:
            return args

    def _pyg_lib_sample(self, data, nodes, **kwargs) -> Data:
        colptr, row, _ = data.adj_t.csr()
        out = torch.ops.pyg.neighbor_sample(
            colptr, row, nodes.to(colptr.dtype),
            self.fanout, None, None, True, # csc
            self.replace, self.directed, False, # not disjoint
            'uniform',  # temporal strategy, no effect
            self.return_eid,
            **kwargs,
            **self.kw,
        )
        #  row, col, node, edge, num_sampled_nodes, num_sampled_edges = out
        return out

    def _torch_sparse_sample(self, data, nodes, **kwargs) -> Data:
        colptr, row, _ = data.adj_t.csr()
        # colptr, row, perm = data.csc_data
        out = torch.ops.torch_sparse.neighbor_sample(
            colptr, row, nodes.to(colptr.dtype),
            self.fanout, self.replace, self.directed,
            **kwargs, **self.kw,
        )
        node, row, col, edge = out
        return row, col, node, edge, None, None

class NodeInducedGraphSampler:
    '''
    Subgraph sampler from the input data based on the provided node batch
    Used in RevGNN mini-batch training
    '''
    def __init__(self, filter_data=filter_data):
        self.filter_data = filter_data

    def __call__(self, data: Data, nodes: torch.Tensor) -> Data:
        return self.subgraph(data, nodes)

    def subgraph(self, data: Data, nodes: torch.Tensor) -> Data:
        ...

class RWGraphSampler(NodeInducedGraphSampler):
    '''
    Subgraph sampler based on random walks from the provided node batch
    Proposed by GraphSAINT
    '''
    def __init__(self, steps: int, filter_data=filter_data):
        super().__init__(self, filter_data)
        self.steps = steps
    
    def __call__(self, data: Data, nodes: torch.Tensor) -> Data:
        rw_nodes = self.rw(data, nodes)
        return self.subgraph(data, rw_nodes)
    
    def rw(self, data, nodes) -> torch.Tensor:
        ...

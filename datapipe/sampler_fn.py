from typing import Callable, Any
import copy
import torch
from torch_geometric.data import Data
from torch_geometric.typing import Tensor, OptTensor
from torch_geometric.loader.utils import filter_data

NodeSampler = Callable[[Data, OptTensor, dict], Any]

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
        unbatch=False,
        **overwrite_kw
    ):
        self.fanout = fanout
        self.replace = replace
        self.directed = directed
        self.return_eid = return_eid
        self.filter_data = filter_data if filter_data_fn is None else filter_data_fn
        self.unbatch = unbatch
        self.kw = overwrite_kw
        try:
            import pyg_lib
            self.sample = self._pyg_lib_sample
        except ImportError:
            self.sample = self._torch_sparse_sample
        self.sample = self._torch_sparse_sample

    def __call__(self, *args, **kwargs) -> Data:
        if self.unbatch:
            data, nodes = args[0][0]
        else:
            data, nodes = args[0]
        sample_out = self.sample(data, nodes, **kwargs)
        row, col, node, edge = sample_out
        # slice node and edge attributes
        data = self.filter_data(data, node, row, col, edge, perm=None)
        data.batch = None
        data.input_id = nodes
        data.batch_size = nodes.size(0)
        return data

    def _pyg_lib_sample(self, data, nodes, **kwargs) -> Data:
        colptr, row, _ = data.adj_t.csr()
        # colptr, row, perm = data.csc_data
        out = torch.ops.pyg.neighbor_sample(
            colptr, row, nodes.to(colptr.dtype),
            self.fanout, None, None, True, # csc
            self.replace, self.directed, False, # not disjoint
            'uniform',  # temporal strategy, no effect
            self.return_eid,
            **kwargs,
            **self.kw,
        )
        row, col, node, edge, *per_hop = out
        return row, col, node, edge

    def _torch_sparse_sample(self, data, nodes, **kwargs) -> Data:
        colptr, row, _ = data.adj_t.csr()
        # colptr, row, perm = data.csc_data
        out = torch.ops.torch_sparse.neighbor_sample(
            colptr, row, nodes.to(colptr.dtype),
            self.fanout, self.replace, self.directed,
            **kwargs, **self.kw,
        )
        node, row, col, edge = out
        return row, col, node, edge

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

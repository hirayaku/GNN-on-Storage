from dataclasses import dataclass
from typing import Union
import torch
import gnnos

@dataclass
class GnnosPartGraph:
    '''
    Partitioned graph in GNNoS
    '''
    num_nodes: int
    psize: int
    part_ptr: Union[torch.Tensor, gnnos.TensorStore]    # P + 1
    src_nids: Union[torch.Tensor, gnnos.TensorStore]        # |V|
    dst_ptr: Union[torch.Tensor, gnnos.TensorStore]         # |V| + 1
    dst_nids: Union[torch.Tensor, gnnos.TensorStore]        # }E|

    def adj(self, node_i):
        start, end = self.dst_ptr[node_i], self.dst_ptr[node_i+1]
        adj = self.dst_nids[start:end]
        return self.src_nids[node_i], adj

    def part_src(self, part_i):
        start, end = self.part_ptr[part_i], self.part_ptr[part_i+1]
        nodes = self.src_nids[start:end]
        return nodes

    def part_dst(self, part_i):
        start, end = self.part_ptr[part_i], self.part_ptr[part_i+1]
        ptr_start, ptr_end = self.dst_ptr[start], self.dst_ptr[end]
        return self.dst_nids[ptr_start:ptr_end]

    def size(self, part_i) -> int:
        return (self.part_ptr[part_i+1] - self.part_ptr[part_i]).item()


# @dataclass
# class GnnosScache:
#     num_nodes: int
#     psize: int
#     # for subgraph induced by src_nids (CSF)
#     src_nids: Union[torch.Tensor, gnnos.TensorStore]
#     dst_ptr: Union[torch.Tensor, gnnos.TensorStore]
#     dst_nids: Union[torch.Tensor, gnnos.TensorStore]
#     # for all other edges (COO)
#     part_ptr: Union[torch.Tensor, gnnos.TensorStore]
#     coo_src: Union[torch.Tensor, gnnos.TensorStore]
#     coo_dst: Union[torch.Tensor, gnnos.TensorStore]

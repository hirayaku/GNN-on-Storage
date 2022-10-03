from dataclasses import dataclass
from typing import Union
import torch
import tqdm
import gnnos

@dataclass
class GnnosPartGraphCOO:
    '''
    Partitioned graph (COO formats) in GNNoS
    '''
    num_nodes: int
    psize: int
    part_ptr: Union[torch.Tensor, gnnos.TensorStore]    # P + 1
    src_nids: Union[torch.Tensor, gnnos.TensorStore]    # |E|
    dst_nids: Union[torch.Tensor, gnnos.TensorStore]    # |E|

    def __post_init__(self):
        self.inmem = isinstance(self.part_ptr, torch.Tensor)

    def slice(self, i):
        return slice(self.part_ptr[i], self.part_ptr[i+1])

    def __getitem__(self, i):
        s = self.slice(i)
        if self.inmem:
            src, dst = self.src_nids[s], self.dst_nids[s]
        else:
            src = self.src_nids.slice(s.start, s.stop).tensor()
            dst = self.dst_nids.slice(s.start, s.stop).tensor()
        return torch.vstack((src, dst))

    def __len__(self):
        return self.psize

    def check_partition(self, assigns: torch.Tensor, which='src'):
        print(f"Checking partitions for {which}")
        for i in tqdm.tqdm(range(self.psize)):
            part = self[i]
            if which == 'src':
                nids = part[0]
            else:
                nids = part[1]
            assert (assigns[nids] == i).all(), f"Partition {i}"
        print("Passed")

@dataclass
class GnnosPartGraph:
    '''
    Partitioned graph (COO formats) in GNNoS
    '''
    num_nodes: int
    psize: int
    part_ptr: Union[torch.Tensor, gnnos.TensorStore]    # P + 1
    edge_index: Union[torch.Tensor, gnnos.COOStore]   # (2,|E|)

    def __post_init__(self):
        self.inmem = isinstance(self.part_ptr, torch.Tensor)

    def slice(self, i):
        return slice(self.part_ptr[i], self.part_ptr[i+1])

    def __getitem__(self, i):
        s = self.slice(i)
        if self.inmem:
            return self.edge_index[:, s]
        else:
            return torch.vstack(self.edge_index.slice(s.start, s.stop).tensor())

    def __len__(self):
        return self.psize

    @property
    def src_nids(self):
        return self.edge_index[0]

    @property
    def dst_nids(self):
        return self.edge_index[1]

    def check_partition(self, assigns: torch.Tensor, which='src'):
        print(f"Checking partitions for {which}")
        for i in tqdm.tqdm(range(self.psize)):
            part = self[i]
            if which == 'src':
                nids = part[0]
            else:
                nids = part[1]
            assert (assigns[nids] == i).all(), f"Partition {i}"
        print("Passed")
    
@dataclass
class GnnosPartGraphCSF:
    '''
    Partitioned graph (CSF formats) in GNNoS
    '''
    num_nodes: int
    psize: int
    part_ptr: Union[torch.Tensor, gnnos.TensorStore]    # P + 1
    src_nids: Union[torch.Tensor, gnnos.TensorStore]    # |V|
    dst_ptr: Union[torch.Tensor, gnnos.TensorStore]     # |V| + 1
    dst_nids: Union[torch.Tensor, gnnos.TensorStore]    # |E|

    def __getitem__(self, part_i):
        start, end = self.part_ptr[part_i], self.part_ptr[part_i+1]
        dst_start, dst_end = self.dst_ptr[start], self.dst_ptr[end]
        dst_ptr = self.dst_ptr[start:end] - self.dst_ptr[start]
        dst_nids = self.dst_nids[dst_start:dst_end]
        return self.src_nids[start:end], dst_ptr, dst_nids

    def __len__(self):
        return self.psize


import os, time
from tqdm import tqdm
import torch
import torch.multiprocessing as torch_mp
from torch_geometric.loader import NeighborLoader
from data.graphloader import NodePropPredDataset

if __name__ == "__main__":
    torch_mp.set_sharing_strategy('file_system')
    dataset = NodePropPredDataset(
        '/mnt/md0/hb_datasets/ogbn_papers100M', mmap=(False, True),
        formats='csc'
    )
    indices = dataset.get_idx_split()
    loader = NeighborLoader(
        dataset[0], input_nodes=indices['valid'][:10],
        batch_size=10,
        num_neighbors=[15,10,5],
        num_workers=12,
    )
    for _ in tqdm(loader):
        pass
    loader = NeighborLoader(
        dataset[0], input_nodes=indices['test'][:10],
        batch_size=10,
        num_neighbors=[15,10,5],
        num_workers=12,
    )
    for _ in tqdm(loader):
        pass
    del dataset, loader

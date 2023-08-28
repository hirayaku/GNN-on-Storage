import os, time
from tqdm import tqdm
import torch
import torch.multiprocessing as torch_mp
import torch.distributed as dist
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from data.graphloader import NodePropPredDataset

print("PyTorch:", torch.__version__, *torch.__path__)
print("CPU parallelism:", torch.get_num_threads())

def load_data(split):
    dataset = NodePropPredDataset('/mnt/md0/hb_datasets/ogbn_products', formats='csc')
    data = dataset[0]
    nids = dataset.get_idx_split(split)
    return data, nids
    # from ogb.nodeproppred import PygNodePropPredDataset
    # dataset = PygNodePropPredDataset('ogbn-products', '/mnt/md0/datasets', transform=T.ToSparseTensor())
    # nids = dataset.get_idx_split()
    # return dataset[0], nids[split]

def run_neighbor_loader():
    data, train_nid = load_data('train')
    train_nid = train_nid[torch.randperm(train_nid.size(0))]
    loader = NeighborLoader(
        data, input_nodes=train_nid,
        batch_size=1000,
        num_neighbors=[15,10,5],
        num_workers=12,
        shuffle=True,
        drop_last=True,
    )
    for e in range(3):
        now = time.time()
        edges, iters = 0, 0
        for batch in tqdm(loader):
            edges += batch.adj_t.nnz()
            iters += 1
        print(f"Epoch {e} done: {time.time() - now:.2f}s. #avg-edges:", edges//iters)

if __name__ == "__main__":
    run_neighbor_loader()

import os, argparse
import torch
from torch_geometric.utils import degree, index_to_mask, mask_to_index
from data.graphloader import NodePropPredDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()
    dataset = args.dataset.replace('-', '_')
    dataset = NodePropPredDataset(
        os.path.join("/mnt/md0/hb_datasets", dataset), mmap=True, formats='csc')
    data = dataset[0]
    train_id = dataset.get_idx_split()['train']
    #  train_id = torch.randperm(dataset.num_nodes)[:dataset.num_nodes//100]
    ptr, _, _ = data.adj_t.csr()
    degrees = ptr[1:]-ptr[:-1]
    print(degrees[train_id].sum().item() / train_id.size(0))

    sg = data.adj_t[train_id, train_id]
    print("training subgraph:")
    print(sg.sizes(), sg.nnz(), sg.nnz()/sg.size(0))

    num_nodes = dataset.num_nodes
    perm = torch.randperm(num_nodes)
    train_mask = index_to_mask(train_id, num_nodes)
    perm_1_8 = perm[:int(num_nodes * (1/8-0.01))]
    perm_1_16 = perm[:int(num_nodes * (1/16-0.01))]

    mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    mask[train_mask] = True
    mask[perm_1_8] = True
    nodes = mask_to_index(mask)
    #  masked_train = mask_to_index(train_mask & mask)
    sg = data.adj_t[train_id, nodes]
    print("1/8 buffer with training nodes:")
    print(sg.sizes(), sg.nnz(), sg.nnz()/sg.size(0))

    mask[:] = False
    mask[train_mask] = True
    mask[perm_1_16] = True
    nodes = mask_to_index(mask)
    #  masked_train = mask_to_index(train_mask & mask)
    sg = data.adj_t[train_id, nodes]
    print("1/16 buffer with training nodes:")
    print(sg.sizes(), sg.nnz(), sg.nnz()/sg.size(0))


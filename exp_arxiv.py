# Peak GPU memory usage is around 1.57 G
# | RevGNN Models           | Test Acc        | Val Acc         |
# |-------------------------|-----------------|-----------------|
# | 112 layers 160 channels | 0.8307 ± 0.0030 | 0.9290 ± 0.0007 |
# | 7 layers 160 channels   | 0.8276 ± 0.0027 | 0.9272 ± 0.0006 |

# imports

import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GroupAddRev, SAGEConv
from torch_geometric.utils import index_to_mask
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import math
import random
import numpy as np

from gat import CustomGAT

import config
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset  # noqa

from torch_geometric.utils import remove_self_loops, add_self_loops

import dgl


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


# adds features!
def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


epsilon = 1 - math.log(2)


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:03d}")

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()

        # Memory-efficient aggregations:
        data = transform(data)

        # giving each an incremental id
        data.n_id = torch.arange(data.train_mask.shape[0]).to(device)

        # randomly create mask subset of train
        ## NOTE: will cause problems if it's not 50%
        rand_label_mask = (
            torch.rand(data.train_mask.shape[0], device=device) < mask_rate
        )
        rand_label_mask_full = data.train_mask & rand_label_mask
        rand_unlabelled_mask_full = data.train_mask & ~rand_label_mask

        # add labels to the randomly created subset of train
        new_inputs = add_labels(data.x, data.y, data.n_id[rand_label_mask_full])

        # get the values which are in train but not in rand_label
        out = model(new_inputs, data.adj_t)

        # get index of unlabelled
        unlabel_idx = data.n_id[~rand_label_mask_full]
        for _ in range(1):
            out = out.detach()
            torch.cuda.empty_cache()
            # update input of unlabelled
            new_inputs[unlabel_idx, -n_classes:] = F.softmax(out[unlabel_idx], dim=-1)
            out = model(new_inputs, data.adj_t)

        # out = model(data.x, data.adj_t)
        # rand_unlabelled_mask_full = data.train_mask

        # calculate loss and take step
        # loss = F.cross_entropy(out[rand_unlabelled_mask_full], data.y[rand_unlabelled_mask_full].view(-1))
        # print(out, data.y)
        # print(out.shape, data.y.view(-1).shape)
        # [nodes, categories]; [nodes]
        loss = custom_loss_function(
            out[rand_unlabelled_mask_full], data.y[rand_unlabelled_mask_full].view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(
            data.train_mask[rand_unlabelled_mask_full].sum()
        )
        total_examples += int(data.train_mask[rand_unlabelled_mask_full].sum())
        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(epoch):
    model.eval()

    y_true = {"train": [], "valid": [], "test": []}
    y_pred = {"train": [], "valid": [], "test": []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:03d}")

    for data in test_loader:
        # Memory-efficient aggregations
        data = transform(data)

        # out = model(data.x, data.adj_t)

        new_inputs = add_labels(data.x, data.y, data.n_id_all[data.train_mask])
        out = model(new_inputs, data.adj_t)  # .argmax(dim=-1, keepdim=True)

        unlabel_idx = data.n_id_all[data.valid_mask | data.test_mask]
        for _ in range(1):
            new_inputs[unlabel_idx, -n_classes:] = F.softmax(out[unlabel_idx], dim=-1)
            out = model(new_inputs, data.adj_t)

        out_eval = out.argmax(dim=-1, keepdim=True)

        for split in ["train", "valid", "test"]:
            mask = data[f"{split}_mask"]
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out_eval[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_acc = evaluator.eval(
        {
            "y_true": torch.cat(y_true["train"], dim=0),
            "y_pred": torch.cat(y_pred["train"], dim=0),
        }
    )["acc"]

    valid_acc = evaluator.eval(
        {
            "y_true": torch.cat(y_true["valid"], dim=0),
            "y_pred": torch.cat(y_pred["valid"], dim=0),
        }
    )["acc"]

    test_acc = evaluator.eval(
        {
            "y_true": torch.cat(y_true["test"], dim=0),
            "y_pred": torch.cat(y_pred["test"], dim=0),
        }
    )["acc"]

    return train_acc, valid_acc, test_acc, out


def main():
    global device, model, train_loader, test_loader, optimizer, transform, mask_rate, n_classes, evaluator

    argparser = argparse.ArgumentParser(
        "GAT implementation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--tensorboard-folder", type=str, default=config.run_dir)
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument(
        "--partition-type", choices=["random", "metis", "fennel"], default="random"
    )
    argparser.add_argument("--partition-num", type=int, default=1)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num-epochs", type=int, default=1000)
    argparser.add_argument("--batch-size", type=int, default=1)
    args = argparser.parse_args()

    run_dir = args.tensorboard_folder

    # set seed
    set_seed(args.seed)

    # lr = 0.002

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([T.ToDevice(device), T.ToSparseTensor()])
    dataset = PygNodePropPredDataset(
        "ogbn-arxiv",  # root,
        # transform=T.ToUndirected())
        transform=T.Compose([T.ToUndirected(), T.AddSelfLoops()]),
    )
    evaluator = Evaluator(name="ogbn-arxiv")

    # dataset.edge_index, _ = remove_self_loops(dataset.edge_index)
    # dataset.edge_index = add_self_loops(dataset.edge_index, num_nodes=dataset.x.size(0))

    data = dataset[0]

    n_classes = dataset.num_classes
    data.n_id_all = torch.arange(data.num_nodes)
    mask_rate = 0.5

    split_idx = dataset.get_idx_split()
    for split in ["train", "valid", "test"]:
        data[f"{split}_mask"] = index_to_mask(split_idx[split], data.y.shape[0])

    num_parts = args.partition_num
    part_type = args.partition_type

    if part_type == "random":
        train_loader = RandomNodeLoader(data, num_parts=num_parts)
    elif part_type == "metis":
        from torch_geometric.loader import ClusterData, ClusterLoader

        cluster_data = ClusterData(data, num_parts=num_parts)
        train_loader = ClusterLoader(cluster_data)
    elif part_type == "fennel":
        from FennelGNN.partitioner import FennelLBPartitioner
        import cluster

        train_nids = dataset.get_idx_split()[
            "train"
        ]  # this partitioner asks for train_nids
        partitioner = FennelLBPartitioner(
            dataset[0], psize=num_parts, train_nids=train_nids
        )
        results = (
            partitioner.partition()
        )  # results is an IntTensor storing the partition id of each node
        cluster_data = cluster.ClusterData(data, cluster=results, num_parts=num_parts)
        train_loader = cluster.ClusterLoader(cluster_data)
    else:
        raise Exception("Error! No valid partition type found!")

    # Increase the num_parts of the test loader if you cannot fit
    # the full batch graph into your GPU:
    test_loader = RandomNodeLoader(data, num_parts=1)

    relu = torch.nn.ReLU()

    final_train_list = []
    final_test_list = []
    best_val_list = []

    # set writer
    writer = SummaryWriter(
        log_dir=f"./{run_dir}/all_tricks_{datetime.now()}_{part_type}_{num_parts}_${args.batch_size}_seed_{args.seed}"
    )
    # reset params
    best_val = 0.0
    final_train = 0.0
    final_test = 0.0
    patience = config.patience
    patience_cnt = config.patience

    model = CustomGAT(
        in_feats=dataset.num_features + dataset.num_classes,
        n_hidden=250,
        n_classes=dataset.num_classes,
        n_layers=3,  # You can try 1000 layers for fun
        dropout=0.75,
        n_heads=3,
        activation=torch.nn.functional.relu,  # None, # 'relu',
        use_symmetric_norm=True,
        input_drop=0.25,
        edge_drop=0.3,
        use_attn_dst=False,
    ).to(device)

    threshold = config.threshold

    # set optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0)

    # epoch
    for epoch in range(1, args.num_epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch)
        loss = train(epoch)
        train_acc, val_acc, test_acc, final_pred = test(epoch)
        if val_acc - best_val < threshold:
            patience_cnt = patience_cnt - 1
        else:
            patience_cnt = patience
        if val_acc > best_val:
            best_val = val_acc
            final_train = train_acc
            final_test = test_acc
        if patience_cnt == 0:
            break
        writer.add_scalar("Train/loss", loss, epoch)
        writer.add_scalar("Train/acc", train_acc, epoch)
        writer.add_scalar("Val/acc", val_acc, epoch)
        writer.add_scalar("Test/acc", test_acc, epoch)

        print(
            f"Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
            f"Test: {test_acc:.4f}"
        )

    print(
        f"Final Train: {final_train:.4f}, Best Val: {best_val:.4f}, "
        f"Final Test: {final_test:.4f}"
    )
    final_train_list.append(final_train)
    final_test_list.append(final_test)
    best_val_list.append(best_val)

    print("Val Accs:", best_val_list)
    print("Test Accs:", final_test_list)
    print(f"Average val accuracy: {np.mean(best_val_list)} ± {np.std(best_val_list)}")
    print(
        f"Average test accuracy: {np.mean(final_test_list)} ± {np.std(final_test_list)}"
    )
    print(f"End of Exp for {part_type} with {num_parts} Partitions")


if __name__ == "__main__":
    main()


import argparse, tqdm, json5
import torch
import torch.nn.functional as F
from models.pyg import gen_model
from data.graphloader import NodePropPredDataset

def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args, _ = parser.parse_known_args()
    with open(args.config) as fp:
        args_dict = json5.load(fp)
    return args_dict

def get_model(in_feats: int, out_feats:int, model_conf: dict) -> torch.nn.Module:
    args = argparse.Namespace()
    for k in model_conf:
        setattr(args, k, model_conf[k])
    return gen_model(in_feats, out_feats, args)

def get_dataset(root: str):
    return NodePropPredDataset(root, mmap=(True, True), formats=('csc',))

def train(model, optimizer, dataloader, device, description='train'):
    model.train()
    minibatches = tqdm.tqdm(dataloader) # prog bar
    minibatches.set_description_str(description)
    num_iters = total_loss = total_correct = total_examples = 0
    mfg_sizes = num_nodes = alive_nodes = batch_score = 0

    for batch in minibatches:
        bsize = batch.batch_size
        dev_attrs = [key for key in batch.keys if not key.endswith('_mask')]
        batch = batch.to(device, *dev_attrs, non_blocking=True)
        optimizer.zero_grad()
        y = batch.y[:bsize].long().view(-1)
        y_hat = model(batch.x, batch.adj_t)[:bsize]
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        # collect stats
        num_iters += 1
        total_loss += float(loss) * bsize
        batch_correct = int((y_hat.argmax(dim=-1) == y).sum())
        total_correct += batch_correct
        total_examples += bsize
        mfg_sizes += batch.adj_t.nnz()
        num_nodes += batch.num_nodes
        if 'alive_nodes' in batch:
            alive_nodes += batch.alive_nodes
        if 'quality_score' in batch:
            batch_score += batch.quality_score

    train_acc = total_correct / total_examples
    return (
        total_loss / total_examples, train_acc,         # loss, acc
        num_nodes / num_iters, alive_nodes / num_iters, # nodes, alive_nodes
        mfg_sizes / num_iters, batch_score / num_iters,
    )

def train_partitioner(model, optimizer, dataloader, device, description='train'):
    model.train()
    minibatches = tqdm.tqdm(dataloader) # prog bar
    minibatches.set_description_str(description)
    num_iters = total_loss = total_correct = total_examples = 0
    mfg_sizes = num_nodes = alive_nodes = batch_score = 0

    for batch in minibatches:
        num_train = len(batch[1])
        # idx = batch[1]
        train_mask = torch.zeros(len(batch[0].y), dtype=torch.bool).to(device)
        train_mask[batch[1]] = 1
        batch = batch[0]
        dev_attrs = [key for key in batch.keys if not key.endswith('_mask')]
        batch = batch.to(device, *dev_attrs, non_blocking=True)
        optimizer.zero_grad()
        y = batch.y[train_mask].long().view(-1)
        y_hat = model(batch.x, batch.adj_t)[train_mask]
        # y = batch.y[idx].long().view(-1)
        # y_hat = model(batch.x, batch.adj_t)[idx]
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        # collect stats
        num_iters += 1
        total_loss += float(loss) * num_train
        batch_correct = int((y_hat.argmax(dim=-1) == y).sum())
        total_correct += batch_correct
        total_examples += num_train
        mfg_sizes += batch.adj_t.nnz()
        num_nodes += batch.num_nodes
        if 'alive_nodes' in batch:
            alive_nodes += batch.alive_nodes
        if 'quality_score' in batch:
            batch_score += batch.quality_score

    train_acc = total_correct / total_examples
    return (
        total_loss / total_examples, train_acc,         # loss, acc
        num_nodes / num_iters, alive_nodes / num_iters, # nodes, alive_nodes
        mfg_sizes / num_iters, batch_score / num_iters,
    )



@torch.no_grad()
def eval_batch(model, dataloader, device, description='eval'):
    model.eval()
    minibatches = tqdm.tqdm(dataloader)
    minibatches.set_description_str(description)
    total_loss = total_correct = total_examples = 0
    for batch in minibatches:
        bsize = batch.batch_size
        dev_attrs = [key for key in batch.keys if not key.endswith('_mask')]
        batch = batch.to(device, *dev_attrs, non_blocking=True)
        y = batch.y[:bsize].long().view(-1)
        y_hat = model(batch.x, batch.adj_t)[:bsize]
        loss = F.nll_loss(y_hat, y)
        # collect stats
        total_loss += float(loss) * bsize
        batch_correct = int((y_hat.argmax(dim=-1) == y).sum())
        total_correct += batch_correct
        total_examples += bsize
    acc = total_correct / total_examples
    return total_loss / total_examples, acc

@torch.no_grad()
def eval_full(model, data, device, masks, description='eval'):
    model.eval()
    y_hat = model(data.x.to(device), data.adj_t.to(device)).cpu()
    def compute_acc(mask):
        y = data.y[mask].long().view(-1)
        loss = float(F.nll_loss(y_hat[mask], y))
        acc = int((y_hat[mask].argmax(dim=-1) == y).sum()) / y.shape[0]
        return loss, acc
    if isinstance(masks, tuple) or isinstance(masks, list):
        return tuple(
            compute_acc(mask) for mask in masks
        )
    else:
        return compute_acc(masks)

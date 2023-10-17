import time, argparse, tqdm, json5
import torch
import torch.nn.functional as F
from models.pyg import gen_model
from data.graphloader import NodePropPredDataset

def get_model(in_feats: int, out_feats:int, model_conf: dict) -> torch.nn.Module:
    args = argparse.Namespace()
    for k in model_conf:
        setattr(args, k, model_conf[k])
    return gen_model(in_feats, out_feats, args)

def get_optimizer(model, params: dict):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    if params.get('lr_schedule', None) == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=params['lr_decay'],
            patience=params['lr_step'],
        )
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler

def get_dataset(root: str, mmap=True, random=False):
    return NodePropPredDataset(root, mmap=mmap, random=random, formats=('csc','coo'))

def train(model, optimizer, dataloader, device, description='train'):
    model.train()
    minibatches = tqdm.tqdm(dataloader)
    minibatches.set_description_str(description)
    num_iters = total_loss = total_correct = total_examples = 0
    mfg_sizes = num_nodes = alive_nodes = batch_score = 0

    start = time.time()
    for batch in minibatches:
        bsize = batch.batch_size
        dev_attrs = [key for key in batch.keys if not key.endswith('_mask')]
        batch = batch.to(device, *dev_attrs, non_blocking=True)
        optimizer.zero_grad()
        y = batch.y[:bsize].long().view(-1)
        y_hat = model(batch.x, batch.edge_index,
                      batch.num_sampled_nodes,
                      batch.num_sampled_edges)[:bsize]
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        # collect stats
        num_iters += 1
        total_loss += float(loss) * bsize
        batch_correct = int((y_hat.argmax(dim=-1) == y).sum())
        total_correct += batch_correct
        total_examples += bsize
        mfg_sizes += batch.edge_index.size(1)
        num_nodes += batch.num_nodes
        if 'alive_nodes' in batch:
            alive_nodes += batch.alive_nodes
        if 'quality_score' in batch:
            batch_score += batch.quality_score
    end = time.time()

    train_acc = total_correct / total_examples
    return (
        total_loss / total_examples, train_acc,         # loss, acc
        num_nodes / num_iters, alive_nodes / num_iters, # nodes, alive_nodes
        mfg_sizes / num_iters, batch_score / num_iters,
        end - start,
    )

import torch.profiler as profiler
from torch.profiler import ProfilerActivity

def train_profile(model, optimizer, dataloader, device, description='train'):
    def trace_handler(prof: profiler.profile):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    model.train()
    minibatches = tqdm.tqdm(dataloader)
    minibatches.set_description_str(description)
    num_iters = total_loss = total_correct = total_examples = 0
    mfg_sizes = num_nodes = alive_nodes = batch_score = 0

    with profiler.profile(
        schedule=profiler.schedule(wait=1, warmup=3, active=1, repeat=1),
        # on_trace_ready=trace_handler,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir/profile"),
        record_shapes=True, with_stack=True,
    ) as p:
        start = time.time()
        for batch in minibatches:
            bsize = batch.batch_size
            dev_attrs = [key for key in batch.keys if not key.endswith('_mask')]
            batch = batch.to(device, *dev_attrs, non_blocking=True)
            optimizer.zero_grad()
            y = batch.y[:bsize].long().view(-1)
            if batch.edge_index is not None:
                y_hat = model(batch.x, batch.edge_index,
                              batch.num_sampled_nodes,
                              batch.num_sampled_edges)[:bsize]
            else:
                y_hat = model(batch.x, batch.adj_t)[:bsize]
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            p.step()
            # collect stats
            num_iters += 1
            total_loss += float(loss) * bsize
            batch_correct = int((y_hat.argmax(dim=-1) == y).sum())
            total_correct += batch_correct
            total_examples += bsize
            mfg_sizes += batch.edge_index.size(1)
            num_nodes += batch.num_nodes
            if 'alive_nodes' in batch:
                alive_nodes += batch.alive_nodes
            if 'quality_score' in batch:
                batch_score += batch.quality_score
        end = time.time()

    train_acc = total_correct / total_examples
    return (
        total_loss / total_examples, train_acc,         # loss, acc
        num_nodes / num_iters, alive_nodes / num_iters, # nodes, alive_nodes
        mfg_sizes / num_iters, batch_score / num_iters,
        end - start,
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
        if batch.edge_index is not None:
            y_hat = model(batch.x, batch.edge_index,
                          batch.num_sampled_nodes,
                          batch.num_sampled_edges)[:bsize]
        else:
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
    y_hat = model(data.x.to(device), data.edge_index.to(device)).cpu()
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


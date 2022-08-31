# This is the baseline GraphSAGE code from GNS repo
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from sklearn.metrics import f1_score
from pyinstrument import Profiler
import os, time
import os.path as osp
from dgl.multiprocessing import shared_tensor
from dgl.utils import pin_memory_inplace, unpin_memory_inplace
import torch.distributed as dist

class SAGE1(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference_batched(self, g, x, idx, device, batch_size, n_workers):
        #  sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([32]*self.n_layers)
        print(idx)
        print(batch_size)
        dataloader = dgl.dataloading.DataLoader(g, idx,
                     sampler, batch_size=batch_size, shuffle=True,
                     drop_last=False, num_workers=n_workers)
        y = th.zeros(g.num_nodes(), self.n_classes)
        #for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        for (input_nodes, output_nodes, blocks) in tqdm.tqdm(dataloader):
            batch_inputs = x[input_nodes].float().to(device)
            blocks = [block.int().to(device) for block in blocks]
            y[output_nodes] = self.forward(blocks, batch_inputs).cpu()
        return y

    def inference(self, g, x, device, batch_size, n_workers=0):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        batch_size : integer, The number of items to collect in a batch.
        Returns: tensor, the predictions for all nodes in the graph.

        The inference could handle any number of nodes and layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(g, th.arange(g.num_nodes()),
                sampler, batch_size=batch_size, shuffle=True, drop_last=False,
                num_workers=n_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]
                block = block.int().to(device)
                h = x[input_nodes].to(device)
                if h.dtype != th.float32:
                    h = h.float()
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[output_nodes] = h.cpu()
            x = y
        return y

class SAGE_DIST(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.n_classes = n_classes
        self.n_hidden = n_hidden

    def _forward_layer(self, l, block, x):
        h = self.layers[l](block, x)
        if l != len(self.layers) - 1:
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self._forward_layer(l, blocks[l], h)
        return h
 
    def inference(self, g, x, device, batch_size, n_workers=0):
        """
        Perform inference in layer-major order rather than batch-major order.
        That is, infer the first layer for the entire graph, and store the
        intermediate values h_0, before infering the second layer to generate
        h_1. This is done for two reasons: 1) it limits the effect of node
        degree on the amount of memory used as it only proccesses 1-hop
        neighbors at a time, and 2) it reduces the total amount of computation
        required as each node is only processed once per layer.
        """
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
                g, th.arange(g.num_nodes(), device=device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=n_workers, use_ddp=True, use_uva=True)

        for l, layer in enumerate(self.layers):
            # in order to prevent running out of GPU memory, we allocate a
            # shared output tensor 'y' in host memory, pin it to allow UVA
            # access from each GPU during forward propagation.
            y = shared_tensor((g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes))
            pin_memory_inplace(y)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader) \
                    if dist.get_rank() == 0 else dataloader:
                if l == 0:
                    h = x[input_nodes].to(device)
                else:
                    h = blocks[0].srcdata['h']
                if h.dtype != th.float32:
                    h = h.float()
                h = self._forward_layer(l, blocks[0], h)
                y[output_nodes] = h.to(y.device)
            # make sure all GPUs are done writing to 'y'
            dist.barrier()
            if l > 0:
                unpin_memory_inplace(g.ndata['h'])
            if l + 1 < len(self.layers):
                # assign the output features of this layer as the new input
                # features for the next layer
                g.ndata['h'] = y
            else:
                # remove the intermediate data from the graph
                g.ndata.pop('h')
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def compute_f1(pred, labels):
    return f1_score(labels, th.argmax(pred, dim=1), average='micro')

def evaluate(model, g, nfeat, labels, val_nid, device, batch_size, num_workers):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        #  pred = model.inference_batched(g, nfeat, val_nid, device, 102400, num_workers)
        pred = model.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    # return compute_f1(pred[val_nid], labels[val_nid])
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


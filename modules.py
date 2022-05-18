import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, mask, device, batch_size=1000, num_workers=0, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata['h'] = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=['h'])
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        if buffer_device is None:
            buffer_device = device

        # TODO: can't hold all the hidden features in memory
        # one approach is to create a partitioning from valid sets.
        # Validation is performed on randomly selected partitions
        for l, layer in enumerate(self.layers[:-1]):
            y = torch.zeros(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device)
            for _, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata['h']
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            g.ndata['h'] = y

        # inference on the last layer: apply masks to save computation
        target_nodes = g.nodes()[mask]
        dataloader_ll = dgl.dataloading.DataLoader(
                g, target_nodes.to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers,
                persistent_workers=(num_workers > 0))
        layer_l = self.layers[-1]
        y = torch.zeros(target_nodes.shape, self.n_classes, device=buffer_device)
        for _, output_nodes, blocks in tqdm.tqdm(dataloader_ll):
            x = blocks[0].srcdata['h']
            h = layer_l(blocks[0], x)
            y[output_nodes] = h.to(buffer_device)

        return y

# TODO: add more modules


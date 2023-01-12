import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

import dgl
import dgl.nn as dglnn
import dgl.function as fn

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 psize, bsize):
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
        self.p_cut = (psize - 1) / (bsize - 1)
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for layer in self.layers:
            layer.apply(weight_reset)

    def _forward_layer(self, l, block, x):
        if self.training and 'cut' in block.edata:
            # cut-edges have weight (p-1)/(k-1), in-edges have weight 1
            block.edata['w'] = block.edata['cut'].float() * (self.p_cut-1)
            block.edata['w'] += torch.ones((block.num_edges(), ), device=block.device)
            block.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'ws'))
            block.ndata['ws'] = block.in_degrees() / block.ndata['ws']
            block.apply_edges(fn.e_mul_v('w', 'ws', 'w'))
            h = self.layers[l](block, x, block.edata['w'])
        else:
            h = self.layers[l](block, x)
        if l != len(self.layers) - 1:
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def forward_mfg(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = self._forward_layer(l, blocks[l], h)
        return h.log_softmax(dim=-1)

    def forward_full(self, g, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = self._forward_layer(l, g, h)
        return h.log_softmax(dim=-1)

    def forward(self, graph, x):
        if isinstance(graph, list):
            return self.forward_mfg(graph, x)
        else:
            return self.forward_full(graph, x)

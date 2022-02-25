import math

import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class GraphSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_lynorm=True):
        super(GraphSAGELayer, self).__init__()
        # The input feature size gets doubled as we concatenated the original
        # features with the new features.
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        with g.local_scope():
            norm = self.get_norm(g)
            if g.is_block:
                g.srcdata['h'] = h
                g.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
                ah = g.dstdata.pop('h')
                h = self.concat(h[:g.num_dst_nodes()], ah, norm)
            else:
                g.ndata['h'] = h
                g.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
                ah = g.ndata.pop('h')
                h = self.concat(h, ah, norm)

        h = self.dropout(h)
        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 full_batch=True):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphSAGELayer(in_feats, n_hidden, activation=activation,
                                          dropout=dropout, use_lynorm=True))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GraphSAGELayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
                               use_lynorm=True))
        # output layer
        self.layers.append(GraphSAGELayer(n_hidden, n_classes, activation=None,
                                          dropout=dropout, use_lynorm=False))

        self.full_batch = full_batch

    def forward(self, g, h):
        if self.full_batch:
            return self.inference(g, h)
        else:
            blocks = g
            for _, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
            return h
    
    def inference(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h

class ModifiedSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_lynorm=True,
                 **kwargs):
        super(ModifiedSAGELayer, self).__init__()
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        g.ndata['h'] = h
        if self.training:
            g.update_all(fn.u_mul_e('h', 'rs', 'm'),
                         fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, g.ndata['norm'])
        else:
            norm = self.get_norm(g)
            g.update_all(fn.copy_src(src='h', out='m'),
                         fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        h = self.dropout(h)
        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class GNNModule(nn.Module):
    def __init__(self,
                 GNNLayer,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GNNModule, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GNNLayer(in_feats, n_hidden, activation=activation,
                                    dropout=dropout,use_lynorm=True))
        # hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(
                GNNLayer(n_hidden, n_hidden, activation=activation, dropout=dropout,
                         use_lynorm=True))
        # output layer
        self.layers.append(GNNLayer(n_hidden, n_classes, activation=None,
                                    dropout=dropout, use_lynorm=False))

    def forward(self, g, h):
        return self.inference(g, h)
    
    def inference(self, g, h):
        if isinstance(g, dgl.DGLGraph):
            for layer in self.layers:
                h = layer(g, h)
        else:
            blocks = g
            for _, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
        return h
    
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl.multiprocessing import shared_tensor

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
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for layer in self.layers:
            layer.apply(weight_reset)

    def _forward_layer(self, l, block, x):
        if 'w' not in block.edata:
            block.edata['w'] = torch.ones((block.num_edges(), )).to(block.device)
        h = self.layers[l](block, x, block.edata['w'])
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

class SAGE_mlp(nn.Module):
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
        self.bns = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        for _ in range(0, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(n_hidden, n_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        def weight_reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        for layer in self.layers:
            layer.apply(weight_reset)
        for bn in self.bns:
            bn.apply(weight_reset)
        self.mlp.apply(weight_reset)

    def _forward_layer(self, l, block, x):
        if 'w' not in block.edata:
            block.edata['w'] = torch.ones((block.num_edges(), )).to(block.device)
        h = self.layers[l](block, x, block.edata['w'])
        h = self.activation(self.bns[l](h))
        return self.dropout(h)

    def forward_mfg(self, blocks, x):
        h = x
        for l, block in enumerate(blocks):
            h = self._forward_layer(l, block, h)
        h = self.mlp(h)
        return h.log_softmax(dim=-1)

    def forward_full(self, g, x):
        h = x
        for l in range(self.n_layers):
            h = self._forward_layer(l, g, h)
        h = self.mlp(h)
        return h.log_softmax(dim=-1)

    def forward(self, graph, x):
        if isinstance(graph, list):
            return self.forward_mfg(graph, x)
        else:
            return self.forward_full(graph, x)

# For SAGE_res_incep
# https://github.com/mengyangniu/ogbn-papers100m-sage
class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 embed_dim,
                 num_layers: int,
                 act: str = 'ReLU',
                 bn: bool = False,
                 end_up_with_fc=False,
                 bias=True):
        super(MLP, self).__init__()
        self.module_list = []
        for i in range(num_layers):
            d_in = input_dim if i == 0 else hidden_dim
            d_out = embed_dim if i == num_layers - 1 else hidden_dim
            self.module_list.append(nn.Linear(d_in, d_out, bias=bias))
            if end_up_with_fc:
                continue
            if bn:
                self.module_list.append(nn.BatchNorm1d(d_out))
            self.module_list.append(getattr(nn, act)(True))
        self.module_list = nn.Sequential(*self.module_list)

    def reset_parameters(self):
        for module in self.module_list:
            if isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()

    def forward(self, x):
        return self.module_list(x)

class SAGE_res_incep(nn.Module):
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
        self.bns = nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Linear(in_feats, n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
            self.res_linears.append(torch.nn.Identity())
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', bias=False, feat_drop=dropout))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.res_linears.append(torch.nn.Identity())
        self.mlp = MLP(in_feats + n_hidden * n_layers, 2 * n_classes, n_classes, num_layers=2, bn=True,
                       end_up_with_fc=True, act='LeakyReLU')
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.profile = locals()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, blocks, x):
        collect = []
        #  h = blocks[0].srcdata['feat']
        h = self.dropout(x)
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            h = self.bns[l](h)
            h = self.activation(h)
            h = self.dropout(h)
            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)
        #  return self.mlp(torch.cat(collect, -1))
        return torch.log_softmax(self.mlp(torch.cat(collect, -1)), dim=-1)



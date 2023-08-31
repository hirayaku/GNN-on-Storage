import torch
import torch.nn as nn
import torch.nn.functional as F

class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

class GATConvPyG(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        use_symmetric_norm=False,
    ):
        from torch_geometric.nn import GATConv, SAGEConv
        super(GATConvPyG, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._use_symmetric_norm = use_symmetric_norm
        self.gat = GATConv(
            self._in_feats, self._out_feats, num_heads,
            negative_slope=negative_slope, attn_drop=attn_drop,
            bias=False
        )
        self.edge_drop = edge_drop
        if residual:
            self.res_fc = nn.Linear(self._in_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        self.gat.reset_parameters()
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    # TODO: perm should be used to help with edge_drop
    def forward(self, feat, edge_index, perm=None):
        from torch_geometric.utils import dropout_adj, degree
        src, dst, _ = edge_index.coo()
        if self._use_symmetric_norm:
            degs = degree(src, edge_index.size(0)).float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        # edge_index_drop, _ = dropout_adj(
        #     edge_index,
        #     p=self.edge_drop,
        #     num_nodes=edge_index.size(0),
        #     training=self.training
        # )
        rst = self.gat(feat, edge_index).view(-1, self._num_heads, self._out_feats)

        if self._use_symmetric_norm:
            degs = degree(dst, edge_index.size(1)).float().clamp(min=1)
            norm = torch.pow(degs, 0.5)
            shp = norm.shape + (1,) * (rst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self._activation is not None:
            rst = self._activation(rst)
        return rst
    

class GATConvDGL(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        use_symmetric_norm=False,
    ):
        from dgl.nn.pytorch.conv import GATConv
        super(GATConvDGL, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._use_symmetric_norm = use_symmetric_norm
        self.gat = GATConv(
            self._in_feats, self._out_feats, num_heads,
            negative_slope=negative_slope, attn_drop=attn_drop,
            bias=False
        )
        self.edge_drop = edge_drop
        if residual:
            self.res_fc = nn.Linear(self._in_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        self.gat.reset_parameters()
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    # TODO: perm should be used to help with edge_drop
    def forward(self, feat, edge_index, perm=None):
        import dgl
        graph = dgl.graph(edge_index.coo()[:2], num_nodes=edge_index.size(0))
        if self._use_symmetric_norm:
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm

        # if self.training and self.edge_drop > 0:
        #     if perm is None:
        #         perm = torch.randperm(graph.number_of_edges(), device=e.device)
        #     bound = int(graph.number_of_edges() * self.edge_drop)
        #     eids = perm[bound:]
        #     graph.edata["a"] = torch.zeros_like(e)
        #     graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
        # else:
        #     graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
        rst = self.gat(graph, feat).view(-1, self._num_heads, self._out_feats)

        if self._use_symmetric_norm:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, 0.5)
            shp = norm.shape + (1,) * (rst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self._activation is not None:
            rst = self._activation(rst)
        return rst
    
class CustomGAT(nn.Module):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        n_heads,
        activation,
        dropout=0.0,
        input_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        use_symmetric_norm=False,
        use_pyg=False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.num_heads = n_heads

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            num_heads = n_heads if i < n_layers - 1 else 1
            out_channels = n_heads

            if use_pyg:
                self.convs.append(
                    GATConvPyG(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        residual=True,
                        activation=F.relu,
                        use_symmetric_norm=use_symmetric_norm,
                    )
                )
            else:
                self.convs.append(
                    GATConvDGL(
                        in_hidden,
                        out_hidden,
                        num_heads=num_heads,
                        attn_drop=attn_drop,
                        edge_drop=edge_drop,
                        residual=True,
                        activation=F.relu,
                        use_symmetric_norm=use_symmetric_norm,
                    )
                )
            if i < n_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(n_classes, weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index):
        h = x
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](h, edge_index)
            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.batch_norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h


import os
import random

import dgl.function as fn
import torch

from partition_utils import *


class ClusterIter(object):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, dn, g, psize, batch_clusters, seed_nid, use_pp=True, return_nodes=False):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_clusters: int
            The number of partitions in one batch
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        """
        self.use_pp = use_pp
        self.g = g.subgraph(seed_nid)

        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        self.batch_clusters = batch_clusters
        print("getting partitioned graph...")
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join('./datasets/', dn + '_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./datasets/', exist_ok=True)
                self.par_li = get_partition_list(self.g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(self.g, psize)
        self.max = int((psize) // batch_clusters)
        random.shuffle(self.par_li)

        if not return_nodes:
            self.get_fn = lambda par_arr, i, psize, batch_size: get_subgraph(self.g, par_arr, i, psize, batch_size)
        else:
            self.get_fn = get_partition_nodes

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['feat']
        with torch.no_grad():
            g.update_all(fn.copy_src(src='feat', out='m'),
                         fn.sum(msg='m', out='feat'),
                         None)
            pre_feats = g.ndata['feat'] * norm
            # use graphsage embedding aggregation style
            g.ndata['feat'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['feat'].device)
        return norm

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.get_fn(self.par_li, self.n, self.psize, self.batch_clusters)
            self.n += 1
            return result
        else:
            random.shuffle(self.par_li)
            raise StopIteration

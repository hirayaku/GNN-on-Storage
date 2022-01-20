import os, time
import random

import numpy as np
from numpy.core.fromnumeric import partition
import torch
import dgl.function as fn

import utils
import partition_utils


class ClusterIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    The sampler either returns a subgraph induced by a batch of
    clusters (default), or the node IDs in the batch (return_nodes=True)
    '''
    def __init__(self, args, g, seed_nid, balance_ntypes=None, return_nodes=False):
        """Initialize the sampler.

        Paramters
        ---------
        args : argparse.Namespace
            Parsed arguments acquired from the command line
        g  : DGLGraph
            The full graph of dataset
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        return_nodes: bool
            Whether to return node IDs or the induced subgraph
        """
        self.dataset_dir = os.path.join(args.rootdir, args.dataset.replace('-', '_'))
        self.partition_dir = os.path.join(self.dataset_dir, 'partition')
        self.use_pp = args.use_pp
        self.g = g if args.semi_supervised else g.subgraph(seed_nid)
        if args.cluster_method == "METIS":
            self.partition_func = lambda g, psize: \
                partition_utils.get_partition_list(g, psize, balance_ntypes)
            print("ClusterIter: METIS node partitioning")
        elif args.cluster_method == "New":
            self.partition_func = partition_utils.get_edge_partition_list
            print("ClusterIter: new partitioning method")
        else:
            self.partition_func = partition_utils.get_rand_partition_list
            print("ClusterIter: random node partitioning")

        # precalc the aggregated features from training graph only
        if self.use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = args.psize
        self.batch_clusters = args.batch_clusters
        print("getting partitioned graph...")
        # cache the partitions of known datasets&partition number
        if args.dataset:
            if not args.balance_train:
                args.cluster_method += "-imb"
            fn = os.path.join(self.partition_dir,
                f'{args.dataset}_{args.cluster_method}_p{self.psize}.npy')
            if os.path.exists(fn):
                par_li = np.load(fn, allow_pickle=True)
                self.par_li = [utils.to_torch_tensor(par) for par in par_li]
            else:
                os.makedirs(self.partition_dir, exist_ok=True)
                self.par_li = self.partition_func(self.g, self.psize)
                np.save(fn, [par.numpy() for par in self.par_li])
        else:
            self.par_li = self.partition_func(self.g, self.psize)
        
        # get the count of appearances for each node
        replicates = torch.zeros(g.num_nodes())
        for par in self.par_li:
            replicates[par] += 1
        g.ndata["count"] = replicates

        self.max = int((self.psize) // self.batch_clusters)
        random.shuffle(self.par_li)

        if not return_nodes:
            self.get_fn = lambda par_arr, i, psize, batch_size: partition_utils.get_subgraph(self.g, par_arr, i, psize, batch_size)
        else:
            self.get_fn = partition_utils.get_partition_nodes

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

class ClusterFeatIter(object):
    '''
    The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    Different from ClusterIter, this iterator returns a tuple of
    nodes and feats each time.
    '''
    def __init__(self, args, g, feats, return_nodes=False):
        self.dataset_dir = os.path.join(args.rootdir, args.dataset.replace('-', '_'))
        self.partition_dir = os.path.join(self.dataset_dir, 'partition')
        assert (not args.use_pp), "ClusterFeatIter doesn't support pre-aggregation"

        # self.g = g.subgraph(seed_nid)
        self.g = g
        self.psize = args.psize
        self.batch_clusters = args.batch_clusters

        print("getting partitioned graph...")
        new_partition = False
        # cache the partitions of known datasets&partition number
        fn = os.path.join(self.partition_dir, args.dataset + '_par_{}.npy'.format(self.psize))
        if os.path.exists(fn):
            par_li = np.load(fn, allow_pickle=True)
            self.par_li = [utils.to_torch_tensor(par) for par in par_li]
        else:
            os.makedirs(self.partition_dir, exist_ok=True)
            self.par_li = partition_utils.get_partition_list(
                self.g, self.psize, self.g.ndata['train_mask'])
            np.save(fn, [par.numpy() for par in self.par_li])
            new_partition = True
        self.par_id = torch.arange(0, self.psize)

        # calculate the starting offsets for features in each partition
        self.par_offsets = torch.cumsum(torch.tensor([0] + [len(par) for par in self.par_li]), dim=0)
        assert self.par_offsets[-1] == feats.shape[0]

        print("shuffling the node features...")
        # reshuffle the node features
        cluster_feat_path = os.path.join(self.partition_dir, f"feat.cluster{args.psize}")
        if not new_partition and os.path.exists(cluster_feat_path):
            self.cluster_feats = np.memmap(cluster_feat_path, mode='r', dtype='float32',
                                           shape=tuple(feats.shape))
        else:
            self.cluster_feats = self.shuffle_features(feats, cluster_feat_path)
        # self.cluster_feats = feats;

        self.max = int(self.psize // self.batch_clusters)

        if not return_nodes:
            self.get_fn = self.get_subgraph
        else:
            self.get_fn = self.get_partition

        # shuffle partitions for randomness
        self.shuffle_iter()

    def shuffle_features(self, feats, cluster_feat_path):
        '''
        given a partition list, input node features,
        generate a shuffled feature file where clustered node features are put adjacent,
        and return a psize+1 array of cluster feature offsets
        '''
        # creates nid -> feat_offsets map, sorted by nid
        feat_offsets = torch.arange(0, feats.shape[0])

        par_array = torch.cat(self.par_li)
        nid_to_feat = torch.stack((par_array, feat_offsets))
        nid_to_feat = nid_to_feat[:, nid_to_feat[0].argsort()]

        cluster_feat_mmap = np.memmap(cluster_feat_path, mode='w+', dtype='float32',
                                      shape=tuple(feats.shape)) # shape must be tuple
        cluster_feat_mmap[nid_to_feat[1], :] = feats[:, :]
        cluster_feat_mmap.flush()
        return cluster_feat_mmap

    def get_partition(self, i):
        par_batch_ids = [self.par_id[s] for s in range(
            i * self.batch_clusters, (i + 1) * self.batch_clusters) if s < self.psize]
        par_batch_nids = [self.par_li[s] for s in par_batch_ids]
        # read features from mmap cluster_feats
        par_feats = [torch.tensor(self.cluster_feats[self.par_offsets[s]:self.par_offsets[s+1],:])
            for s in par_batch_ids]
        return torch.cat(par_batch_nids), torch.cat(par_feats)

    def get_subgraph(self, i):
        nids, feats = self.get_partition(i)

        # NOTE: nodes in the induced subgraph will keep the order of nids
        # therefore we don't need to reorder feats
        subgraph = self.g.subgraph(nids)

        return subgraph, feats

    def shuffle_iter(self):
        self.par_id = self.par_id[torch.randperm(self.par_id.shape[0])]

    def __len__(self):
        return self.max

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.max:
            result = self.get_fn(self.n)
            self.n += 1
            return result        # self.par_id = torch.arange(0, self.psize)
        else:
            self.shuffle_iter()
            raise StopIteration

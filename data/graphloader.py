import os, os.path as osp, copy
from typing import Optional, Tuple, Union, Dict
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from utils import sort, mem_usage
from data.datasets import serialize
from data.io import TensorMeta, TensorType, DtypeDecoder, MmapTensor
from data.io import load_tensor, madvise, MADV_OPTIONS
import logging
logger = logging.getLogger()

def partition_dir(root, method, parts):
    return os.path.join(root, f"{method}-P{parts}")

def pivots_dir(root, method, parts):
    return os.path.join(root, f"{method}-P{parts}-pivots")

class NodePropPredDataset(object):
    '''
    Mmap-friendly dataset for node predictions.\n
    Loosely follows the interface of NodePropPredDataset in ogb.
    '''
    def __init__(
        self, dataset_dir:str,
        mmap: Union[bool, Tuple[bool, bool], Dict[str, bool]]=False,
        random: Union[bool, Tuple[bool, bool], Dict[str, bool]]=False,
        formats: Union[Tuple, str]=('coo', 'csr', 'csc'),
        create_formats:bool=False,
    ):
        self.root = dataset_dir
        with open(osp.join(self.root, 'metadata.json')) as f_meta:
            self.meta_info = DtypeDecoder(root=self.root).decode(f_meta.read())

        self.attr_dict = self.meta_info.get('attr', dict())
        self.data_dict = self.meta_info.get('data', dict())
        self.idx_dict = self.meta_info.get('idx', dict())
        for attr in self.attr_dict:
            setattr(self, attr, self.meta_info['attr'][attr])
        # whether to load in memory or mmap
        if isinstance(mmap, tuple) or isinstance(mmap, list):
            self.mmap = {'graph': mmap[0], 'feat': mmap[1]}
        elif isinstance(mmap, Dict):
            self.mmap = copy.copy(mmap)
        else:
            self.mmap = {'graph': mmap, 'feat': mmap}
        # whether to optimize for random IO
        if isinstance(random, tuple) or isinstance(random, list):
            self.random = {'graph': random[0], 'feat': random[1]}
        elif isinstance(random, Dict):
            pass
        else:
            self.random = {'graph': random, 'feat': random}
        # which formats to load
        if isinstance(formats, str):
            formats = (formats,)
        self.formats = {fmt: False for fmt in formats}
        self.create_formats = create_formats

        super(NodePropPredDataset, self).__init__()
        self._mmaps = set()
        self.load_data()

    @property
    def graph_mode(self):
        return (
            TensorType.MmapTensor if self.mmap['graph'] else TensorType.PlainTensor,
            self.random['graph']
        )
    @property
    def feat_mode(self):
        return (
            TensorType.MmapTensor if self.mmap['feat'] else TensorType.PlainTensor,
            self.random['feat']
        )

    def tensor_from_meta(self, meta: TensorMeta, mode=(TensorType.PlainTensor, False)):
        # type_lookup = {
        #     'mem': TensorType.PlainTensor,
        #     'shm': TensorType.ShmemTensor,
        #     'shmem': TensorType.ShmemTensor,
        #     'mmap': TensorType.MmapTensor,
        #     'ext': TensorType.MmapTensor,
        #     'disk': TensorType.DiskTensor,
        #     'remote': TensorType.RemoteTensor
        # }
        ttype, random = mode
        meta.random = random
        tensor = load_tensor(meta, ttype)
        if ttype == TensorType.MmapTensor:
            self._mmaps.add((tensor, mode))
        return tensor

    def fill_mmaps(self):
        '''
        restore the operating mode of mmaped tensors
        '''
        for tensor, mode in self._mmaps:
            if mode[1] is True:
                madvise(
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                    MADV_OPTIONS.MADV_RANDOM
                )
            else:
                madvise(
                    tensor.data_ptr(),
                    tensor.numel() * tensor.element_size(),
                    MADV_OPTIONS.MADV_NORMAL
                )

    def drop_mmaps(self):
        '''
        immediately free all mmaped tensors in the current dataloader
        '''
        for tensor, _ in self._mmaps:
            madvise(
                tensor.data_ptr(),
                tensor.numel() * tensor.element_size(),
                MADV_OPTIONS.MADV_DONTNEED
            )

    def _load_graph(self):
        graph_dicts = self.data_dict['graph']
        num_nodes = self.data_dict['num_nodes']
        self.data.num_nodes = num_nodes
        coo_sorted = None
        formats_to_load = dict(**self.formats)
        for graph in graph_dicts:
            edge_format = graph['format']
            if edge_format in formats_to_load:
                if edge_format == 'coo':
                    coo_sorted = graph.get('sorted', None)
                    src = self.tensor_from_meta(graph['edge_index'][0], self.graph_mode)
                    dst = self.tensor_from_meta(graph['edge_index'][1], self.graph_mode)
                    self.data.edge_index = (src, dst)
                    self.formats[edge_format] = True
                elif edge_format == 'csc':
                    ptr = self.tensor_from_meta(graph['adj_t'][0])
                    ids = self.tensor_from_meta(graph['adj_t'][1], self.graph_mode)
                    self.data.adj_t = SparseTensor(rowptr=ptr, col=ids, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
                    self.formats[edge_format] = True
                elif edge_format == 'csr':
                    ptr = self.tensor_from_meta(graph['adj'][0])
                    ids = self.tensor_from_meta(graph['adj'][1], self.graph_mode)
                    self.data.adj = SparseTensor(rowptr=ptr, col=ids, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
                    self.formats[edge_format] = True
            else:
                logger.info(f"Skipping graph data of format \"{edge_format}\"")

        # Generate CSC/CSR if they don't exist but are required
        need_csc = 'csc' in self.formats and self.formats['csc'] is False
        need_csr = 'csr' in self.formats and self.formats['csr'] is False
        if (need_csc or need_csr) and self.create_formats:
            if 'coo' not in self.formats or self.formats['coo'] is False:
                raise ValueError(f"COO graph not found in {self.root}")
            if self.num_nodes >= (2**63)**0.5:
                raise RuntimeError("Can't handle graph with more than 2^31.5 nodes yet")
            if need_csc:
                # TODO external sort
                src, dst = self.data.edge_index
                to_sort = MmapTensor(TensorMeta.like(dst).temp_())
                to_sort.copy_(dst)
                to_sort *= self.num_nodes
                to_sort += src
                # radix_sort from pyg_lib crashes for papers, fall back to torch.sort
                _, indices = torch.sort(to_sort)
                del _, to_sort
                s_src = src[indices]
                s_dst = dst[indices]
                del indices
                self.data.edge_index = (s_src, s_dst)
                coo_sorted = 'dst'
                ptr = torch.ops.torch_sparse.ind2ptr(s_dst, num_nodes)
                self.data.adj_t = SparseTensor(rowptr=ptr, col=s_src,
                    sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
                if need_csr and not self.is_directed:
                    self.data.adj = self.data.adj_t
                    need_csr = False
            if need_csr:
                to_sort = MmapTensor(TensorMeta.like(src).temp_())
                to_sort = src * self.num_nodes
                to_sort += dst
                _, indices = torch.sort(to_sort)
                del _, to_sort
                s_src = src[indices]
                s_dst = dst[indices]
                del indices
                ptr = torch.ops.torch_sparse.ind2ptr(s_src, num_nodes)
                self.data.adj = SparseTensor(rowptr=ptr, col=s_dst,
                    sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
            logger.info("Serializing graph data in new formats...")
            self.data_dict['graph'] = [
                {'format': 'coo', 'edge_index': self.data.edge_index, 'sorted': coo_sorted},
                {'format': 'csc', 'adj_t': self.data.adj_t.csr()},
                {'format': 'csr', 'adj': self.data.adj.csr()},
            ]
            self.meta_info = serialize({
                'attr': self.attr_dict,
                'data': self.data_dict,
                'idx': self.idx_dict,
                }, self.root,
            )

    def _load_labels(self):
        labels = self.tensor_from_meta(self.data_dict['labels'], mode=self.feat_mode)
        self.data.put_tensor(labels, attr_name='y', index=None)

    def _load_node_feat(self):
        node_feat = self.tensor_from_meta(self.data_dict['node_feat'], mode=self.feat_mode)
        self.data.put_tensor(node_feat, attr_name='x', index=None)
 
    def load_data(self):
        self.data = Data()
        self._load_graph()
        logger.debug("Dataset edge index loaded: {:.2f} MB".format(mem_usage()))
        self._load_labels()
        self._load_node_feat()
        # self._load_edge_feat()
        logger.debug("Dataset fully loaded: {:.2f} MB".format(mem_usage()))
 
    def get_idx_split(self, split=None):
        idx_dict = self.idx_dict
        if split is None:
            train_idx = self.tensor_from_meta(idx_dict['train'])
            valid_idx = self.tensor_from_meta(idx_dict['valid'])
            test_idx = self.tensor_from_meta(idx_dict['test'])
            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        else:
            return self.tensor_from_meta(idx_dict[split])

    def __getitem__(self, idx) -> Data:
        if idx > 0:
            raise IndexError('This dataset has only one graph')
        return self.data
    
    def __len__(self):
        return 1

class ChunkedNodePropPredDataset(NodePropPredDataset):
    def __init__(self, dataset_dir, mmap={'graph': True, 'feat': True}, **kwargs):
        super().__init__(dataset_dir, mmap=mmap, formats=('coo',), **kwargs)

    # overwrite _load_* methods
    def _load_graph(self):
        if 'graph' in self.data_dict:
            num_nodes = self.data_dict['num_nodes']
            self.data.num_nodes = num_nodes
            for graph in self.data_dict['graph']:
                assert graph['format'] == 'coo'
                src = self.tensor_from_meta(graph['edge_index'][0], self.graph_mode)
                dst = self.tensor_from_meta(graph['edge_index'][1], self.graph_mode)
                off = self.tensor_from_meta(graph['edge_index'][2])
                label = graph.get('label', 'edge_index')
                self.data[label] = (src, dst, off)

    def _load_node_feat(self):
        node_feat = self.tensor_from_meta(self.data_dict['node_feat'], mode=self.feat_mode)
        self.data.put_tensor(node_feat, attr_name='x', index=None)
        if self.feat_mode == 'mmap':
            # optimize sequential access on feature data
            node_feat = self.data.x
            madvise(node_feat.data_ptr(), node_feat.numel() * node_feat.element_size(),
                    MADV_OPTIONS.MADV_SEQUENTIAL)

    def _load_labels(self):
        for k, v in self.data_dict.items():
            if k not in ('graph', 'node_feat', 'edge_feat', 'labels'):
                if isinstance(v, TensorMeta):
                    tensor = self.tensor_from_meta(v, self.feat_mode)
                    setattr(self, k, tensor)
                else:
                    setattr(self, k, v)
            elif k == 'labels':
                super()._load_labels()

    def relabel(self, nids: torch.Tensor):
        try:
            relabel_map = self.relabel_map
        except AttributeError:
            relabel_map = torch.empty_like(self.node_map)
            relabel_map[self.node_map] = torch.arange(0, self.num_nodes)
            self.relabel_map = relabel_map
        finally:
            return relabel_map[nids]

    def get_idx_split(self, split=None):
        idx_dict = self.idx_dict
        train_idx = self.tensor_from_meta(idx_dict['train'])
        valid_idx = self.tensor_from_meta(idx_dict['valid'])
        test_idx = self.tensor_from_meta(idx_dict['test'])
        self.splits = {
            'train': self.relabel(train_idx),
            'valid': self.relabel(valid_idx),
            'test':  self.relabel(test_idx)
        }
        if split is None:
            return self.splits
        else:
            return self.splits[split]

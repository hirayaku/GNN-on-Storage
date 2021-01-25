"""Data loaders"""

from collections.abc import Mapping
from abc import ABC, abstractproperty, abstractmethod
import numpy as np
from .. import transform
from ..base import NID, EID
from .. import backend as F
from .. import utils
from ..convert import heterograph
from ..distributed.dist_graph import DistGraph

# pylint: disable=unused-argument
def assign_block_eids(block, frontier):
    """Assigns edge IDs from the original graph to the block.
    See also
    --------
    BlockSampler
    """
    for etype in block.canonical_etypes:
        block.edges[etype].data[EID] = frontier.edges[etype].data[EID][
            block.edges[etype].data[EID]]
    return block

def _tensor_or_dict_to_numpy(ids):
    if isinstance(ids, Mapping):
        return {k: F.zerocopy_to_numpy(v) for k, v in ids.items()}
    else:
        return F.zerocopy_to_numpy(ids)

def _locate_eids_to_exclude(frontier_parent_eids, exclude_eids):
    """Find the edges whose IDs in parent graph appeared in exclude_eids.
    Note that both arguments are numpy arrays or numpy dicts.
    """
    if isinstance(frontier_parent_eids, Mapping):
        result = {
            k: np.isin(frontier_parent_eids[k], exclude_eids[k]).nonzero()[0]
            for k in frontier_parent_eids.keys() if k in exclude_eids.keys()}
        return {k: F.zerocopy_from_numpy(v) for k, v in result.items()}
    else:
        result = np.isin(frontier_parent_eids, exclude_eids).nonzero()[0]
        return F.zerocopy_from_numpy(result)

def _find_exclude_eids_with_reverse_id(g, eids, reverse_eid_map):
    if isinstance(eids, Mapping):
        eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
        exclude_eids = {
            k: F.cat([v, F.gather_row(reverse_eid_map[k], v)], 0)
            for k, v in eids.items()}
    else:
        exclude_eids = F.cat([eids, F.gather_row(reverse_eid_map, eids)], 0)
    return exclude_eids

def _find_exclude_eids_with_reverse_types(g, eids, reverse_etype_map):
    exclude_eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
    reverse_etype_map = {
        g.to_canonical_etype(k): g.to_canonical_etype(v)
        for k, v in reverse_etype_map.items()}
    exclude_eids.update({reverse_etype_map[k]: v for k, v in exclude_eids.items()})
    return exclude_eids

def _find_exclude_eids(g, exclude_mode, eids, **kwargs):
    """Find all edge IDs to exclude according to :attr:`exclude_mode`.
    Parameters
    ----------
    g : DGLGraph
        The graph.
    exclude_mode : str, optional
        Can be either of the following,
        None (default)
            Does not exclude any edge.
        'reverse_id'
            Exclude all edges specified in ``eids``, as well as their reverse edges
            of the same edge type.
            The mapping from each edge ID to its reverse edge ID is specified in
            the keyword argument ``reverse_eid_map``.
            This mode assumes that the reverse of an edge with ID ``e`` and type
            ``etype`` will have ID ``reverse_eid_map[e]`` and type ``etype``.
        'reverse_types'
            Exclude all edges specified in ``eids``, as well as their reverse
            edges of the corresponding edge types.
            The mapping from each edge type to its reverse edge type is specified
            in the keyword argument ``reverse_etype_map``.
            This mode assumes that the reverse of an edge with ID ``e`` and type ``etype``
            will have ID ``e`` and type ``reverse_etype_map[etype]``.
    eids : Tensor or dict[etype, Tensor]
        The edge IDs.
    reverse_eid_map : Tensor or dict[etype, Tensor]
        The mapping from edge ID to its reverse edge ID.
    reverse_etype_map : dict[etype, etype]
        The mapping from edge etype to its reverse edge type.
    """
    if exclude_mode is None:
        return None
    elif exclude_mode == 'reverse_id':
        return _find_exclude_eids_with_reverse_id(g, eids, kwargs['reverse_eid_map'])
    elif exclude_mode == 'reverse_types':
        return _find_exclude_eids_with_reverse_types(g, eids, kwargs['reverse_etype_map'])
    else:
        raise ValueError('unsupported mode {}'.format(exclude_mode))


class BlockSampler(object):
    """Abstract class specifying the neighborhood sampling strategy for DGL data loaders.
    The main method for BlockSampler is :meth:`sample_blocks`,
    which generates a list of blocks for a multi-layer GNN given a set of seed nodes to
    have their outputs computed.
    The default implementation of :meth:`sample_blocks` is
    to repeat :attr:`num_layers` times the following procedure from the last layer to the first
    layer:
    * Obtain a frontier.  The frontier is defined as a graph with the same nodes as the
      original graph but only the edges involved in message passing on the current layer.
      Customizable via :meth:`sample_frontier`.
    * Optionally, if the task is link prediction or edge classfication, remove edges
      connecting training node pairs.  If the graph is undirected, also remove the
      reverse edges.  This is controlled by the argument :attr:`exclude_eids` in
      :meth:`sample_blocks` method.
    * Convert the frontier into a block.
    * Optionally assign the IDs of the edges in the original graph selected in the first step
      to the block, controlled by the argument ``return_eids`` in
      :meth:`sample_blocks` method.
    * Prepend the block to the block list to be returned.
    All subclasses should override :meth:`sample_frontier`
    method while specifying the number of layers to sample in :attr:`num_layers` argument.
    Parameters
    ----------
    num_layers : int
        The number of layers to sample.
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the block.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.
    Notes
    -----
    For the concept of frontiers and blocks, please refer to User Guide Section 6 [TODO].
    """
    def __init__(self, num_layers, return_eids):
        self.num_layers = num_layers
        self.return_eids = return_eids

    def sample_frontier(self, block_id, g, seed_nodes):
        """Generate the frontier given the output nodes.
        The subclasses should override this function.
        Parameters
        ----------
        block_id : int
            Represents which GNN layer the frontier is generated for.
        g : DGLGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The output nodes by node type.
            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        Returns
        -------
        DGLGraph
            The frontier generated for the current layer.
        Notes
        -----
        For the concept of frontiers and blocks, please refer to User Guide Section 6 [TODO].
        """
        raise NotImplementedError

    def _convert_exclude_eids(self, exclude_eids):
        return _tensor_or_dict_to_numpy(exclude_eids) if exclude_eids is not None else None

    def _exclude_eids(self, frontier, exclude_eids):
        """Remove the specific edge IDs from the given frontier.
        """
        if exclude_eids is not None:
            parent_eids = frontier.edata[EID]
            parent_eids_np = _tensor_or_dict_to_numpy(parent_eids)
            located_eids = _locate_eids_to_exclude(parent_eids_np, exclude_eids)
            if not isinstance(located_eids, Mapping):
                # (BarclayII) If frontier already has a EID field and located_eids is empty,
                # the returned graph will keep EID intact.  Otherwise, EID will change
                # to the mapping from the new graph to the old frontier.
                # So we need to test if located_eids is empty, and do the remapping ourselves.
                if len(located_eids) > 0:
                    frontier = transform.remove_edges(frontier, located_eids, store_ids=True)
                    frontier.edata[EID] = F.gather_row(parent_eids, frontier.edata[EID])
            else:
                # (BarclayII) remove_edges only accepts removing one type of edges,
                # so I need to keep track of the edge IDs left one by one.
                new_eids = parent_eids.copy()
                for k, v in located_eids.items():
                    if len(v) > 0:
                        frontier = transform.remove_edges(frontier, v, etype=k, store_ids=True)
                        new_eids[k] = F.gather_row(parent_eids[k], frontier.edges[k].data[EID])
                frontier.edata[EID] = new_eids
        return frontier

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """Generate the a list of blocks given the output nodes.
        Parameters
        ----------
        g : DGLGraph
            The original graph.
        seed_nodes : Tensor or dict[ntype, Tensor]
            The output nodes by node type.
            If the graph only has one node type, one can just specify a single tensor
            of node IDs.
        exclude_eids : Tensor or dict[etype, Tensor]
            The edges to exclude from computation dependency.
        Returns
        -------
        list[DGLGraph]
            The blocks generated for computing the multi-layer GNN output.
        Notes
        -----
        For the concept of frontiers and blocks, please refer to User Guide Section 6 [TODO].
        """
        blocks = []
        exclude_eids = self._convert_exclude_eids(exclude_eids)
        for block_id in reversed(range(self.num_layers)):
            frontier = self.sample_frontier(block_id, g, seed_nodes)

            # Removing edges from the frontier for link prediction training falls
            # into the category of frontier postprocessing
            frontier = self._exclude_eids(frontier, exclude_eids)

            block = transform.to_block(frontier, seed_nodes)

            if self.return_eids:
                assign_block_eids(block, frontier)

            if len(g.ntypes) > 1:
                seed_nodes = {ntype: block.srcnodes[ntype].data[NID] for ntype in block.srctypes}
            else:
                seed_nodes = block.srcdata[NID]

            # Pre-generate CSR format so that it can be used in training directly
            block.create_formats_()
            blocks.insert(0, block)
        return blocks

class Collator(ABC):
    """Abstract DGL collator for training GNNs on downstream tasks stochastically.
    Provides a :attr:`dataset` object containing the collection of all nodes or edges,
    as well as a :attr:`collate` method that combines a set of items from
    :attr:`dataset` and obtains the blocks.
    Notes
    -----
    For the concept of blocks, please refer to User Guide Section 6 [TODO].
    """
    @abstractproperty
    def dataset(self):
        """Returns the dataset object of the collator."""
        raise NotImplementedError

    @abstractmethod
    def collate(self, items):
        """Combines the items from the dataset object and obtains the list of blocks.
        Parameters
        ----------
        items : list[str, int]
            The list of node or edge IDs or type-ID pairs.
        Notes
        -----
        For the concept of blocks, please refer to User Guide Section 6 [TODO].
        """
        raise NotImplementedError

class NodeCollator(Collator):
    """DGL collator to combine nodes and their computation dependencies within a minibatch for
    training node classification or regression on a single graph with neighborhood sampling.
    Parameters
    ----------
    g : DGLGraph
        The graph.
    nids : Tensor or dict[ntype, Tensor]
        The node set to compute outputs.
    block_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors (assume
    the backend is PyTorch):
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(input_nodes, output_nodes, blocks)
    """
    def __init__(self, g, nids, block_sampler):
        self.g = g
        self._is_distributed = isinstance(g, DistGraph)
        if not isinstance(nids, Mapping):
            assert len(g.ntypes) == 1, \
                "nids should be a dict of node type and ids for graph with multiple node types"
        self.nids = nids
        self.block_sampler = block_sampler

        if isinstance(nids, Mapping):
            self._dataset = utils.FlattenedDict(nids)
        else:
            self._dataset = nids

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """Find the list of blocks necessary for computing the representation of given
        nodes for a node classification/regression task.
        Parameters
        ----------
        items : list[int] or list[tuple[str, int]]
            Either a list of node IDs (for homogeneous graphs), or a list of node type-ID
            pairs (for heterogeneous graphs).
        Returns
        -------
        input_nodes : Tensor or dict[ntype, Tensor]
            The input nodes necessary for computation in this minibatch.
            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        output_nodes : Tensor or dict[ntype, Tensor]
            The nodes whose representations are to be computed in this minibatch.
            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        blocks : list[DGLGraph]
            The list of blocks necessary for computing the representation.
        """
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)

        # TODO(BarclayII) Because DistGraph doesn't have idtype and device implemented,
        # this function does not work.  I'm again skipping this step as a workaround.
        # We need to fix this.
        if not self._is_distributed:
            if isinstance(items, dict):
                items = utils.prepare_tensor_dict(self.g, items, 'items')
            else:
                items = utils.prepare_tensor(self.g, items, 'items')
        blocks = self.block_sampler.sample_blocks(self.g, items)
        output_nodes = blocks[-1].dstdata[NID]
        input_nodes = blocks[0].srcdata[NID]

        return input_nodes, output_nodes, blocks

class EdgeCollator(Collator):
    """DGL collator to combine edges and their computation dependencies within a minibatch for
    training edge classification, edge regression, or link prediction on a single graph
    with neighborhood sampling.
    Given a set of edges, the collate function will yield
    * A tensor of input nodes necessary for computing the representation on edges, or
      a dictionary of node type names and such tensors.
    * A subgraph that contains only the edges in the minibatch and their incident nodes.
      Note that the graph has an identical metagraph with the original graph.
    * If a negative sampler is given, another graph that contains the "negative edges",
      connecting the source and destination nodes yielded from the given negative sampler.
    * A list of blocks necessary for computing the representation of the incident nodes
      of the edges in the minibatch.
    Parameters
    ----------
    g : DGLGraph
        The graph from which the edges are iterated in minibatches and the subgraphs
        are generated.
    eids : Tensor or dict[etype, Tensor]
        The edge set in graph :attr:`g` to compute outputs.
    block_sampler : dgl.dataloading.BlockSampler
        The neighborhood sampler.
    g_sampling : DGLGraph, optional
        The graph where neighborhood sampling and message passing is performed.
        Note that this is not necessarily the same as :attr:`g`.
        If None, assume to be the same as :attr:`g`.
    exclude : str, optional
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are
        * None, which excludes nothing.
        * ``'reverse_id'``, which excludes the reverse edges of the sampled edges.  The said
          reverse edges have the same edge type as the sampled edges.  Only works
          on edge types whose source node type is the same as its destination node type.
        * ``'reverse_types'``, which excludes the reverse edges of the sampled edges.  The
          said reverse edges have different edge types from the sampled edges.
        If ``g_sampling`` is given, ``exclude`` is ignored and will be always ``None``.
    reverse_eids : Tensor or dict[etype, Tensor], optional
        The mapping from original edge ID to its reverse edge ID.
        Required and only used when ``exclude`` is set to ``reverse_id``.
        For heterogeneous graph this will be a dict of edge type and edge IDs.  Note that
        only the edge types whose source node type is the same as destination node type
        are needed.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the edge type to its reverse edge type.
        Required and only used when ``exclude`` is set to ``reverse_types``.
    negative_sampler : callable, optional
        The negative sampler.  Can be omitted if no negative sampling is needed.
        The negative sampler must be a callable that takes in the following arguments:
        * The original (heterogeneous) graph.
        * The ID array of sampled edges in the minibatch, or the dictionary of edge
          types and ID array of sampled edges in the minibatch if the graph is
          heterogeneous.
        It should return
        * A pair of source and destination node ID arrays as negative samples,
          or a dictionary of edge types and such pairs if the graph is heterogenenous.
        A set of builtin negative samplers are provided in
        :ref:`the negative sampling module <api-dataloading-negative-sampling>`.
    Examples
    --------
    The following example shows how to train a 3-layer GNN for edge classification on a
    set of edges ``train_eid`` on a homogeneous undirected graph.  Each node takes
    messages from all neighbors.
    Say that you have an array of source node IDs ``src`` and another array of destination
    node IDs ``dst``.  One can make it bidirectional by adding another set of edges
    that connects from ``dst`` to ``src``:
    >>> g = dgl.graph((torch.cat([src, dst]), torch.cat([dst, src])))
    One can then know that the ID difference of an edge and its reverse edge is ``|E|``,
    where ``|E|`` is the length of your source/destination array.  The reverse edge
    mapping can be obtained by
    >>> E = len(src)
    >>> reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])
    Note that the sampled edges as well as their reverse edges are removed from
    computation dependencies of the incident nodes.  This is a common trick to avoid
    information leakage.
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)
    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` on a
    homogeneous graph where each node takes messages from all neighbors (assume the
    backend is PyTorch), with 5 uniformly chosen negative samples per edge:
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_id',
    ...     reverse_eids=reverse_eids, negative_sampler=neg_sampler,
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)
    For heterogeneous graphs, the reverse of an edge may have a different edge type
    from the original edge.  For instance, consider that you have an array of
    user-item clicks, representated by a user array ``user`` and an item array ``item``.
    You may want to build a heterogeneous graph with a user-click-item relation and an
    item-clicked-by-user relation.
    >>> g = dgl.heterograph({
    ...     ('user', 'click', 'item'): (user, item),
    ...     ('item', 'clicked-by', 'user'): (item, user)})
    To train a 3-layer GNN for edge classification on a set of edges ``train_eid`` with
    type ``click``, you can write
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, {'click': train_eid}, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'})
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pair_graph, blocks in dataloader:
    ...     train_on(input_nodes, pair_graph, blocks)
    To train a 3-layer GNN for link prediction on a set of edges ``train_eid`` with type
    ``click``, you can write
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    >>> collator = dgl.dataloading.EdgeCollator(
    ...     g, train_eid, sampler, exclude='reverse_types',
    ...     reverse_etypes={'click': 'clicked-by', 'clicked-by': 'click'},
    ...     negative_sampler=neg_sampler)
    >>> dataloader = torch.utils.data.DataLoader(
    ...     collator.dataset, collate_fn=collator.collate,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, pos_pair_graph, neg_pair_graph, blocks in dataloader:
    ...     train_on(input_nodse, pair_graph, neg_pair_graph, blocks)
    """
    def __init__(self, g, eids, block_sampler, g_sampling=None, exclude=None,
                 reverse_eids=None, reverse_etypes=None, negative_sampler=None):
        self.g = g
        if not isinstance(eids, Mapping):
            assert len(g.etypes) == 1, \
                "eids should be a dict of etype and ids for graph with multiple etypes"
        self.eids = eids
        self.block_sampler = block_sampler

        # One may wish to iterate over the edges in one graph while perform sampling in
        # another graph.  This may be the case for iterating over validation and test
        # edge set while perform neighborhood sampling on the graph formed by only
        # the training edge set.
        # See GCMC for an example usage.
        if g_sampling is not None:
            self.g_sampling = g_sampling
            self.exclude = None
        else:
            self.g_sampling = self.g
            self.exclude = exclude

        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes
        self.negative_sampler = negative_sampler

        if isinstance(eids, Mapping):
            self._dataset = utils.FlattenedDict(eids)
        else:
            self._dataset = eids

    @property
    def dataset(self):
        return self._dataset

    def _collate(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
            items = utils.prepare_tensor_dict(self.g_sampling, items, 'items')
        else:
            items = utils.prepare_tensor(self.g_sampling, items, 'items')

        pair_graph = self.g.edge_subgraph(items)
        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes)

        blocks = self.block_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids)
        input_nodes = blocks[0].srcdata[NID]

        return input_nodes, pair_graph, blocks

    def _collate_with_negative_sampling(self, items):
        if isinstance(items[0], tuple):
            # returns a list of pairs: group them by node types into a dict
            items = utils.group_as_dict(items)
            items = utils.prepare_tensor_dict(self.g_sampling, items, 'items')
        else:
            items = utils.prepare_tensor(self.g_sampling, items, 'items')

        pair_graph = self.g.edge_subgraph(items, preserve_nodes=True)
        induced_edges = pair_graph.edata[EID]

        neg_srcdst = self.negative_sampler(self.g, items)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.g.etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (F.tensor([], dtype), F.tensor([], dtype)))
            for etype in self.g.canonical_etypes}
        neg_pair_graph = heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})

        pair_graph, neg_pair_graph = transform.compact_graphs([pair_graph, neg_pair_graph])
        pair_graph.edata[EID] = induced_edges

        seed_nodes = pair_graph.ndata[NID]

        exclude_eids = _find_exclude_eids(
            self.g,
            self.exclude,
            items,
            reverse_eid_map=self.reverse_eids,
            reverse_etype_map=self.reverse_etypes)

        blocks = self.block_sampler.sample_blocks(
            self.g_sampling, seed_nodes, exclude_eids=exclude_eids)
        input_nodes = blocks[0].srcdata[NID]

        return input_nodes, pair_graph, neg_pair_graph, blocks

    def collate(self, items):
        """Combines the sampled edges into a minibatch for edge classification, edge
        regression, and link prediction tasks.
        Parameters
        ----------
        items : list[int] or list[tuple[str, int]]
            Either a list of edge IDs (for homogeneous graphs), or a list of edge type-ID
            pairs (for heterogeneous graphs).
        Returns
        -------
        Either ``(input_nodes, pair_graph, blocks)``, or
        ``(input_nodes, pair_graph, negative_pair_graph, blocks)`` if negative sampling is
        enabled.
        input_nodes : Tensor or dict[ntype, Tensor]
            The input nodes necessary for computation in this minibatch.
            If the original graph has multiple node types, return a dictionary of
            node type names and node ID tensors.  Otherwise, return a single tensor.
        pair_graph : DGLGraph
            The graph that contains only the edges in the minibatch as well as their incident
            nodes.
            Note that the metagraph of this graph will be identical to that of the original
            graph.
        negative_pair_graph : DGLGraph
            The graph that contains only the edges connecting the source and destination nodes
            yielded from the given negative sampler, if negative sampling is enabled.
            Note that the metagraph of this graph will be identical to that of the original
            graph.
        blocks : list[DGLGraph]
            The list of blocks necessary for computing the representation of the edges.
        """
        if self.negative_sampler is None:
            return self._collate(items)
        else:
            return self._collate_with_negative_sampling(items)
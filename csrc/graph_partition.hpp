#pragma once

#include <memory>
#include <torch/torch.h>
#include "graph_io.hpp"

namespace gnnos {

class NodePartitions {
private:
    struct NodePartitionsData {
        torch::Tensor assignments;  // num_nodes
        torch::Tensor clusters;     // num_nodes
        torch::Tensor cluster_pos;  // psize + 1
        ~NodePartitionsData() {
            LOG(WARNING) << "De-allocate NodePartitionsData";
        }
    };

    std::shared_ptr<NodePartitionsData> obj;

public:
    int psize;

    long num_nodes() const { return obj->assignments.numel(); }

    // return the assignments tensor
    const torch::Tensor &assignments() {
        return obj->assignments;
    }

    // return the node partition at [idx]
    torch::Tensor operator[](int idx) const;

    // create a NodePartitions object from node assignments tensor
    static NodePartitions New(int psize, torch::Tensor assignments);
};

NodePartitions random_partition(const CSRStore &graph, int psize);
NodePartitions random_partition(const COOStore &graph, int psize);
NodePartitions go_assignment(const CSRStore &graph, int psize);


// Blocked COO Store
class BCOOStore: public COOStore {
public:
    BCOOStore() = default;
    // partition by the src node
    static BCOOStore PartitionFrom1D(const COOStore &, NodePartitions);
    // partition by (src, dst)
    static BCOOStore PartitionFrom2D(const COOStore &, NodePartitions);

    // constant methods
    int psize() const { return partition.psize; }
    int num_blocks() const { return edge_pos_.size(); }
    std::vector<long> edge_pos() const { return edge_pos_; }

    // return an edge block as a COOStore
    COOStore coo_block(int blkid);
    // return a subgraph induced by the specified vertex clusters (2D partition only)
    COOArrays cluster_subgraph(const std::vector<int> &cluster_ids);

private:

    // TODO: an alternative design choice is to keep a vector of COOStore,
    // rather than inheritating COOStore and keeping a vector of edge position offsets.
    // It could support a more general form of BCOOStore: COO blocks could come from
    // different files instead of a unified COOStore.
    std::vector<long> edge_pos_;

    // current node partition
    NodePartitions partition;

    // build a BCOOStore on top of COOStore
    // edge_pos represents the edge blocks from COO partitioning
    BCOOStore(COOStore coo, std::vector<long> edge_pos, NodePartitions partition)
        : COOStore(std::move(coo)), edge_pos_(std::move(edge_pos))
        , partition(std::move(partition))
    {
        CHECK_EQ(coo.num_nodes(), partition.num_nodes())
            << "BCOOStore: invalid partition assignments";
    }

};

// Blocked CSR Store
struct BCSRStore {
    BCSRStore() = default;

    static BCSRStore PartitionFrom1D(const CSRStore &, const torch::Tensor &, int psize);
    static BCSRStore PartitionFrom2D(const CSRStore &, const torch::Tensor &, int psize);

    std::vector<CSRStore> blocks;
    NodePartitions partition;
};


}   // ns gnnos

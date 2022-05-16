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
        // ~NodePartitionsData() {
        //     LOG(INFO) << "De-allocate NodePartitionsData";
        // }
    };

    std::shared_ptr<NodePartitionsData> obj;

public:
    int psize;

    long num_nodes() const { return obj->assignments.numel(); }

    // return the assignments tensor
    const torch::Tensor &assignments() {
        return obj->assignments;
    }

    // size of cluster[idx]
    long size(int idx) const;

    // get cluster[idx]
    torch::Tensor operator[](int idx) const;

    // create a NodePartitions object from node assignments tensor
    static NodePartitions New(int psize, torch::Tensor assignments);
};

NodePartitions random_partition(const CSRStore &graph, int psize);
NodePartitions random_partition(const COOStore &graph, int psize);
NodePartitions go_partition(const CSRStore &graph, int psize);


// Blocked COO Store
class BCOOStore: public COOStore {
public:
    BCOOStore() = default;
    // partition by the src node
    static BCOOStore PartitionFrom1D(const COOStore &, NodePartitions);
    // partition by (src, dst)
    static BCOOStore PartitionFrom2D(const COOStore &, NodePartitions);
    // partition by (src, dst)
    template <typename PtrT, typename IdxT>
    static BCOOStore PartitionFrom2D(const CSRStore &, NodePartitions);

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
        TORCH_CHECK(coo.num_nodes() == this->partition.num_nodes(),
            "Invalid partition assignments: ", this->partition.num_nodes());
    }

};

template <typename PtrT, typename IdxT>
BCOOStore BCOOStore::PartitionFrom2D(const CSRStore &csr, NodePartitions partition) {
    auto assigns_vec = partition.assignments().accessor<int, 1>();
    auto psize = partition.psize;
    auto ptr_accessor = csr.ptr_store.accessor<PtrT>();
    auto idx_accessor = csr.idx_store.accessor<IdxT>();

    auto p_counts_ = torch::zeros({psize, psize}, torch::TensorOptions(torch::kLong));
    auto p_counts = p_counts_.accessor<long, 2>();
    constexpr long BLOCK_NODES = 1024;

    // count edges in each partition
    #pragma omp parallel for
    for (long i = 0; i < csr.num_nodes(); i += BLOCK_NODES) {
        auto start = i;
        auto end = i + BLOCK_NODES;
        if (end > csr.num_nodes()) end = csr.num_nodes();
        for (long src = start; src < end; ++src) {
            auto src_blk = assigns_vec[src];
            auto adj = idx_accessor.slice(ptr_accessor[src], ptr_accessor[src+1]);
            for (auto dst : adj) {
                auto dst_blk = assigns_vec[dst];
                #pragma omp atomic
                p_counts[src_blk][dst_blk]++;
            }
        }
    }
    CHECK_EQ(torch::sum(p_counts_).item<long>(), csr.num_edges());
    LOG(INFO) << "Counting complete";

    // compute pos array
    auto pos_ = torch::zeros({psize*psize+1}, p_counts_.options());
    auto pos_1 = pos_.index({torch::indexing::Slice(1, torch::indexing::None)});
    torch::cumsum_out(pos_1, p_counts_.reshape({psize*psize}), 0);
    CHECK_EQ(pos_.index({0}).item<long>(), 0);
    CHECK_EQ(pos_.index({-1}).item<long>(), csr.num_edges());

    // create a unnamed COOStore (int64_t) to place the partitioned graph
    TensorInfo info = TensorOptions(TMPDIR).itemsize(8).shape({csr.num_edges()});
    auto bcoo = COOStore(
        TensorStore::CreateTemp(info),
        TensorStore::CreateTemp(info),
        csr.num_nodes()
    );
    auto bcoo_accessor = bcoo.accessor<long>();

    // shuffle edges into each partition
    {
    // the starting position of each partition
    auto pos_current_ = pos_.clone();
    auto pos_current = pos_current_.accessor<long, 1>();
    // write buffers to increase write granularity
    constexpr int BUF_EDGES=256; // 256 * 8 * 2 = 4kB / partition
    std::vector<std::vector<int64_t>> src_buffers(psize * psize);
    std::vector<std::vector<int64_t>> dst_buffers(psize * psize);
    for (int i = 0; i < psize * psize; ++i) {
        src_buffers[i].reserve(BUF_EDGES);
        dst_buffers[i].reserve(BUF_EDGES);
    }
    // coarsened locks
    std::vector<std::mutex> mutexes(16 * 16);
    #pragma omp parallel for
    for (long i = 0; i < csr.num_nodes(); i += BLOCK_NODES) {
        auto start = i;
        auto end = i + BLOCK_NODES;
        if (end > csr.num_nodes()) end = csr.num_nodes();

        thread_local std::vector<long> src_copy, dst_copy;
        long reserved_pos;
        for (long src = start; src < end; ++src) {
            auto src_blk = assigns_vec[src];
            auto adj = idx_accessor.slice(ptr_accessor[src], ptr_accessor[src+1]);
            for (auto dst : adj) {
                auto dst_blk = assigns_vec[dst];
                auto blkid = src_blk * psize + dst_blk;
                auto lockid = src_blk * 16 / psize * 16 + dst_blk * 16 / psize;
                bool writeout = false;
                {
                    std::lock_guard<std::mutex> guard(mutexes[lockid]);
                    src_buffers[blkid].push_back(src);
                    dst_buffers[blkid].push_back(dst);
                    if (src_buffers[blkid].size() == BUF_EDGES) {
                        writeout = true;
                        // swap out buffers to write out
                        std::swap(src_copy,src_buffers[blkid]);
                        std::swap(dst_copy,dst_buffers[blkid]);
                        reserved_pos = pos_current[blkid];
                        pos_current[blkid] += BUF_EDGES;
                    }
                }
                if (writeout) {
                    bcoo_accessor.slice_put(src_copy.data(), dst_copy.data(),
                        reserved_pos, reserved_pos + BUF_EDGES);
                    src_copy.clear(); dst_copy.clear();
                }
            }
        }
    }

    // write out remaining data
    for (int i = 0; i < psize * psize; ++i) {
        if (src_buffers[i].size() != 0) {
            bcoo_accessor.slice_put(src_buffers[i].data(), dst_buffers[i].data(),
                pos_current[i], pos_current[i] + src_buffers[i].size());
            pos_current[i] += src_buffers[i].size();
        }
    }
    for (int i = 0; i < psize * psize; ++i) {
        CHECK_EQ(pos_current[i], pos_[i+1].item<long>());
    }
    LOG(INFO) << "Partition complete";
    }   // release src_buffers, dst_buffers, pos_current


    long *pos_data = pos_.contiguous().data_ptr<long>();
    std::vector<long> pos(pos_data, pos_data + psize*psize + 1);
    return BCOOStore(bcoo, std::move(pos), std::move(partition));
}

/*
// Blocked CSR Store
struct BCSRStore {
    BCSRStore() = default;

    // partition by the src node
    static BCSRStore PartitionFrom1D(const CSRStore &, NodePartitions);
    // partition by (src, dst)
    static BCSRStore PartitionFrom2D(const CSRStore &, NodePartitions);

    std::vector<CSRStore> blocks;
    NodePartitions partition;
};
*/

}   // ns gnnos

#include "graph_partition.hpp"
#include "utils.hpp"

namespace gnnos {

torch::Tensor NodePartitions::operator[](int idx) const {
    auto start = obj->cluster_pos[idx].item<long>();
    auto end = obj->cluster_pos[idx+1].item<long>();
    return obj->clusters.index({torch::indexing::Slice(start, end)});
}

NodePartitions NodePartitions::New(int psize, torch::Tensor assignments) {
    auto assigns_vec = assignments.accessor<long, 1>();
    auto pos = torch::zeros({psize+1}, torch::dtype(torch::kLong));
    auto pos_vec = pos.accessor<long, 1>();
    for (long i = 0; i < assigns_vec.size(0); ++i) {
        pos_vec[assigns_vec[i]+1]++;
    }
    torch::cumsum(pos, 0);

    // generate clusters tensor
    auto clusters = torch::empty_like(assignments, torch::dtype(torch::kLong));
    auto clusters_vec = clusters.data_ptr<long>();
    auto current_pos = pos.clone();
    auto current_vec = current_pos.data_ptr<long>();
    for (long i = 0; i < assigns_vec.size(0); ++i) {
        clusters_vec[current_vec[assigns_vec[i]]++] = i;
    }

    NodePartitions partition;
    partition.psize = psize;
    partition.obj = std::shared_ptr<NodePartitionsData>(
        new NodePartitionsData{assignments, clusters, pos}
    );

    return partition;
}

static torch::Tensor random_assignment(long num_nodes, int psize) {
    return torch::randint(psize, {num_nodes}, torch::TensorOptions(torch::kInt));
}

NodePartitions random_partition(const COOStore &graph, int psize) {
    return NodePartitions::New(psize, random_assignment(graph.num_nodes(), psize));
}
NodePartitions random_partition(const CSRStore &graph, int psize) {
    return NodePartitions::New(psize, random_assignment(graph.num_nodes(), psize));
}

NodePartitions go_partition(const CSRStore &graph, int psize) {
    auto rand_partitions = random_partition(graph, psize);
    // BCOOStore rand_dcoo = BCOOStore::PartitionFrom1D(coo, rand_assigns, psize);
    return rand_partitions;
}

// BCOOStore methods
BCOOStore BCOOStore::PartitionFrom1D(const COOStore &coo, NodePartitions partition) {
    auto assigns_vec = partition.assignments().accessor<int, 1>();
    auto psize = partition.psize;
    auto edge_accessor = coo.accessor<int64_t>();
    auto p_counts_ = torch::zeros({psize}, torch::TensorOptions(torch::kLong));
    auto p_counts = p_counts_.accessor<long, 1>();
    constexpr long BLOCK_EDGES = 1024 * 1024;

    // count edges in each partition
    #pragma omp parallel for
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src, dst;
        std::tie(src, dst) = edge_accessor.slice(start, end);
        for (long i = 0; i < end - start; ++i) {
            auto src_blk = assigns_vec[src[i]];
            #pragma omp atomic
            p_counts[src_blk]++;
        }
    }
    CHECK_EQ(torch::sum(p_counts_).item<long>(), coo.num_edges());
    LOG(WARNING) << "Counting complete";

    // compute pos array
    auto pos_ = torch::zeros({psize+1}, p_counts_.options());
    auto pos_1 = pos_.index({torch::indexing::Slice(1, torch::indexing::None)});
    torch::cumsum_out(pos_1, p_counts_, 0);
    CHECK_EQ(pos_.index({0}).item<long>(), 0);
    CHECK_EQ(pos_.index({-1}).item<long>(), coo.num_edges());

    // create a unnamed COOStore to place the partitioned graph
    size_t num_nodes;
    TensorInfo src_info, dst_info;
    std::tie(num_nodes, src_info, dst_info) = coo.metadata();
    auto coo_copy = COOStore(
        TensorStore::CreateTemp(src_info.path(TMPDIR).offset(0)),
        TensorStore::CreateTemp(dst_info.path(TMPDIR).offset(0)),
        num_nodes
    );
    auto edge_copy_accessor = coo_copy.accessor<int64_t>();

    // shuffle edges into each partition
    // the starting position of each partition
    auto pos_current_ = pos_.clone();
    auto pos_current = pos_current_.accessor<long, 1>();
    {
    // write buffers to increase write granularity
    constexpr int BUF_EDGES=512;
    int num_threads = torch::get_num_threads();
    std::vector<std::vector<int64_t>> src_buffers(psize * num_threads); // [tid][blkid]
    std::vector<std::vector<int64_t>> dst_buffers(psize * num_threads);
    for (int i = 0; i < psize * num_threads; ++i) {
        src_buffers[i].reserve(BUF_EDGES);
        dst_buffers[i].reserve(BUF_EDGES);
    }
    // coarsened locks
    std::vector<std::mutex> mutexes(64);
    #pragma omp parallel for
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src, dst;
        std::tie(src, dst) = edge_accessor.slice(start, end);

        int tid = torch::get_thread_num();
        long reserved_pos;
        for (long i = 0; i < end - start; ++i) {
            auto src_blk = assigns_vec[src[i]];
            auto bufid = psize * tid + src_blk; 
            src_buffers[bufid].push_back(src[i]);
            dst_buffers[bufid].push_back(dst[i]);

            if (src_buffers[bufid].size() == BUF_EDGES) {
                auto lockid = src_blk * 64 / psize;
                {
                    std::lock_guard<std::mutex> guard(mutexes[lockid]);
                    reserved_pos = pos_current[src_blk];
                    pos_current[src_blk] += BUF_EDGES;
                }
                edge_copy_accessor.slice_put(
                    src_buffers[bufid].data(), dst_buffers[bufid].data(),
                    reserved_pos, reserved_pos + BUF_EDGES);
                src_buffers[bufid].clear();
                src_buffers[bufid].clear();
            }
        }
    }
    // write out remaining data
    for (int tid = 0; tid < num_threads; ++tid) {
        for (int i = 0; i < psize; ++i) {
            int bufid = tid * psize + i;
            if (src_buffers[bufid].size() != 0) {
                edge_copy_accessor.slice_put(
                    src_buffers[bufid].data(), dst_buffers[bufid].data(),
                    pos_current[i], pos_current[i] + src_buffers[bufid].size());
                pos_current[i] += src_buffers[bufid].size();
            }
        }
    }
    }   // release src_buffers, dst_buffers

    for (int i = 0; i < psize; ++i) {
        CHECK_EQ(pos_current[i], pos_[i+1].item<long>());
    }
    LOG(WARNING) << "Partition complete";

    long *pos_data = pos_.contiguous().data_ptr<long>();
    std::vector<long> pos(pos_data, pos_data + psize + 1);
    return BCOOStore(coo_copy, std::move(pos), std::move(partition));
}

BCOOStore BCOOStore::PartitionFrom2D(const COOStore &coo, NodePartitions partition) {
    auto assigns_vec = partition.assignments().accessor<int, 1>();
    auto psize = partition.psize;
    auto edge_accessor = coo.accessor<int64_t>();
    auto p_counts_ = torch::zeros({psize, psize}, torch::TensorOptions(torch::kLong));
    auto p_counts = p_counts_.accessor<long, 2>();
    constexpr long BLOCK_EDGES = 1024 * 1024;

    // count edges in each partition
    #pragma omp parallel for
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src, dst;
        std::tie(src, dst) = edge_accessor.slice(start, end);
        for (long i = 0; i < end - start; ++i) {
            auto src_blk = assigns_vec[src[i]];
            auto dst_blk = assigns_vec[dst[i]];
            #pragma omp atomic
            p_counts[src_blk][dst_blk]++;
        }
    }
    CHECK_EQ(torch::sum(p_counts_).item<long>(), coo.num_edges());
    LOG(WARNING) << "Counting complete";

    // compute pos array
    auto pos_ = torch::zeros({psize*psize+1}, p_counts_.options());
    auto pos_1 = pos_.index({torch::indexing::Slice(1, torch::indexing::None)});
    torch::cumsum_out(pos_1, p_counts_.reshape({psize*psize}), 0);
    CHECK_EQ(pos_.index({0}).item<long>(), 0);
    CHECK_EQ(pos_.index({-1}).item<long>(), coo.num_edges());

    // create a unnamed COOStore to place the partitioned graph
    size_t num_nodes;
    TensorInfo src_info, dst_info;
    std::tie(num_nodes, src_info, dst_info) = coo.metadata();
    auto coo_copy = COOStore(
        TensorStore::CreateTemp(src_info.path(TMPDIR).offset(0)),
        TensorStore::CreateTemp(dst_info.path(TMPDIR).offset(0)),
        num_nodes
    );
    auto edge_copy_accessor = coo_copy.accessor<int64_t>();

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
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src, dst;
        std::tie(src, dst) = edge_accessor.slice(start, end);

        thread_local std::vector<int64_t> src_copy, dst_copy;
        long reserved_pos;
        for (long i = 0; i < end - start; ++i) {
            bool writeout = false;
            auto src_blk = assigns_vec[src[i]];
            auto dst_blk = assigns_vec[dst[i]];
            auto blkid = src_blk * psize + dst_blk;
            auto lockid = src_blk * 16 / psize * 16 + dst_blk * 16 / psize;
            {
                std::lock_guard<std::mutex> guard(mutexes[lockid]);
                src_buffers[blkid].push_back(src[i]);
                dst_buffers[blkid].push_back(dst[i]);
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
                edge_copy_accessor.slice_put(src_copy.data(), dst_copy.data(),
                    reserved_pos, reserved_pos + BUF_EDGES);
                src_copy.clear(); dst_copy.clear();
            }
        }
    }
    // write out remaining data
    for (int i = 0; i < psize * psize; ++i) {
        if (src_buffers[i].size() != 0) {
            edge_copy_accessor.slice_put(src_buffers[i].data(), dst_buffers[i].data(),
                pos_current[i], pos_current[i] + src_buffers[i].size());
            pos_current[i] += src_buffers[i].size();
        }
    }
    #ifndef NDEBUG
    for (int i = 0; i < psize * psize; ++i) {
        CHECK_EQ(pos_current[i], pos_[i+1].item<long>());
    }
    #endif
    LOG(WARNING) << "Partition complete";
    }   // release src_buffers, dst_buffers, pos_current


    long *pos_data = pos_.contiguous().data_ptr<long>();
    std::vector<long> pos(pos_data, pos_data + psize*psize + 1);
    return BCOOStore(coo_copy, std::move(pos), std::move(partition));
}

COOStore BCOOStore::coo_block(int blkid) {
    CHECK_LT(blkid, edge_pos_.size()-1);
    return this->slice(edge_pos_[blkid], edge_pos_[blkid+1]);
}

COOArrays BCOOStore::cluster_subgraph(const std::vector<int> &cluster_ids) {
    int nclusters = cluster_ids.size();
    // create node id mapping
    std::unordered_map<long, long> node_map(num_nodes_ / psize() * nclusters);
    long new_id = 0;
    for (auto cid : cluster_ids) {
        auto cluster = partition[cid];
        for (const auto old_id : tensor_iter<long>(cluster)) {
            node_map.insert({old_id, new_id});
            ++new_id;
        }
    }


    // collect subgraph info
    std::vector<long> blk_pos(nclusters * nclusters + 1);
    blk_pos[0] = 0;
    for (int i = 0; i < nclusters; ++i) {
        auto from_cid = cluster_ids[i];
        for (int j = 0; j < nclusters; ++j) {
            auto to_cid = cluster_ids[j];
            auto blkid = from_cid * nclusters + to_cid;
            auto sg_blkid = i * nclusters + j;
            blk_pos[sg_blkid+1] = edge_pos_[blkid+1] - edge_pos_[blkid];
        }
    }
    for (size_t i = 1; i < blk_pos.size(); ++i) {
        blk_pos[i] += blk_pos[i-1];
    }
    long sg_num_edges = blk_pos[nclusters*nclusters];

    // create COO tensors
    torch::Tensor sg_src_ = torch::empty(
        {sg_num_edges}, torch::TensorOptions(torch::kLong));
    torch::Tensor sg_dst_ = torch::empty_like(sg_src_);
    long *sg_src = sg_src_.data_ptr<long>();
    long *sg_dst = sg_dst_.data_ptr<long>();

    // map coo blocks and fill into tensors
    #pragma omp parallel for
    for (int i = 0; i < nclusters; ++i) {
        auto from_cid = cluster_ids[i];
        for (int j = 0; j < nclusters; ++j) {
            auto to_cid = cluster_ids[j];
            auto blkid = from_cid * nclusters + to_cid;
            auto sg_blkid = i * nclusters + j;

            auto coo = coo_block(blkid);
            CHECK_EQ(coo.num_edges(), blk_pos[sg_blkid+1]-blk_pos[sg_blkid]);
            std::vector<long> src, dst;
            std::tie(src, dst) = coo.accessor<long>().slice(0, coo.num_edges());

            for (auto &src_id : src) src_id = node_map.at(src_id);
            for (auto &dst_id : dst) dst_id = node_map.at(dst_id);
            std::copy(src.begin(), src.end(), sg_src + blk_pos[sg_blkid]);
            std::copy(dst.begin(), dst.end(), sg_dst + blk_pos[sg_blkid]);
        }
    }

    return {std::move(sg_src_), std::move(sg_dst_)};
}

/*
BCSRStore BCSRStore::PartitionFrom1D(const CSRStore &csr,
    const torch::Tensor &assigns, int psize) {
    
    auto assigns_vec = assigns.accessor<int, 1>();
    auto ptr_accessor = csr.ptr_store.accessor<long>();
    auto idx_accessor = csr.idx_store.accessor<long>();
    auto p_counts_ = torch::zeros({psize}, torch::TensorOptions(torch::kLong));
    auto p_counts = p_counts_.accessor<long, 1>();
    std::vector<std::vector<long>> clusters;

    // count edges in each partition
    auto ptr_vec = ptr_accessor.slice(0, csr.num_nodes()+1);
    for (long i = 0; i < csr.num_nodes(); i++) {
        long degree = ptr_vec[i+1] - ptr_vec[i];
        auto blkid = assigns_vec[i];
        clusters[blkid].push_back(i);
        p_counts[blkid] += degree;
    }
    CHECK_EQ(torch::sum(p_counts_).item<long>(), csr.num_edges());
    LOG(WARNING) << "Counting complete";

    // create CSR blocks
    std::vector<CSRStore> csr_blocks;
    for (int i = 0; i < psize; ++i) {
        long p_num_nodes = clusters[i].size();
        long p_num_edges = p_counts[i];
        csr_blocks.push_back({
            TensorStore::CreateTemp(TensorOptions(TMPDIR).itemsize(8).shape({p_num_nodes+1})),
            TensorStore::CreateTemp(TensorOptions(TMPDIR).itemsize(8).shape({p_num_edges}))
        });
    }

    // map edges into csr blocks
    #pragma omp parallel for
    for (long nid = 0; nid < csr.num_nodes(); ++nid) {
        long start = ptr_vec[nid];
        long deg = ptr_vec[nid+1] - ptr_vec[nid];
        auto blkid = assigns_vec[nid];
        auto &blk = csr_blocks[blkid];
        blk.ptr_store[]
    }

}
*/

}

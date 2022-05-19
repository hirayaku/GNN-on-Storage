#include <vector>
#include <tuple>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <torch/torch.h>
#include "graph_io.hpp"

namespace gnnos {

// COOStore methods

COOArrays COOStore::tensor() const {
    return {src_store_.tensor(), dst_store_.tensor()};
}

COOStore COOStore::clone(std::string path, bool fill) {
    size_t src_nbytes = src_store_.numel() * src_store_.itemsize();
    auto new_src_store = TensorStore::Create(
        src_store_.metadata().offset(0).path(path));
    auto new_dst_store = TensorStore::Open(
        dst_store_.metadata().offset(src_nbytes).path(path));
    if (fill) {
        src_store_.copy_to(new_src_store);
        dst_store_.copy_to(new_dst_store);
    }
    return COOStore(new_src_store, new_dst_store, num_nodes());
}

COOStore COOStore::slice(long start, long end) const {
    return COOStore(
        this->src_store_.slice(start, end),
        this->dst_store_.slice(start, end),
        this->num_nodes());
}

// CSRStore methods
CSRStore CSRStore::NewFrom(const COOStore &coo) {
    long num_nodes;
    TensorInfo dst_info;
    std::tie(num_nodes, std::ignore, dst_info) = coo.metadata();

    auto ptr_store = TensorStore::CreateTemp(
        TensorOptions(TMPDIR).shape({num_nodes+1}).dtype(torch::kLong).offset(0));
    auto idx_store = TensorStore::CreateTemp(dst_info.path(TMPDIR).offset(0));

    // count degrees of each node (id starting from 1)
    auto d_counts_ = torch::zeros({num_nodes+1}, torch::dtype(torch::kLong));
    auto d_counts = d_counts_.accessor<long, 1>();
    auto edge_accessor = coo.accessor<long>();
    constexpr long BLOCK_EDGES = 1024 * 1024;

    #pragma omp parallel for
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src;
        std::tie(src, std::ignore) = edge_accessor.slice(start, end);
        for (long i = 0; i < end - start; ++i) {
            auto src_id = src[i];
            #pragma omp atomic
            d_counts[src_id+1] += 1;
        }
    }
    CHECK_EQ(torch::sum(d_counts_).item<long>(), coo.num_edges());
    LOG(INFO) << "Counting complete";

    // compute rowptr array
    auto ptr_tensor_ = torch::cumsum(d_counts_, 0);
    CHECK_EQ(ptr_tensor_[0].item<long>(), 0);
    CHECK_EQ(ptr_tensor_[num_nodes].item<long>(), coo.num_edges());
    ptr_store.accessor<long>().slice_put(ptr_tensor_.data_ptr<long>(), 0, num_nodes+1);

    // generate colidx array
    {
    // starting position of idx store
    auto ptr_current_ = ptr_tensor_.clone();
    auto ptr_current = ptr_current_.accessor<long, 1>();
    auto idx_accessor = idx_store.accessor<long>();
    // coarsened locks
    std::vector<std::mutex> mutexes(64);
    #pragma omp parallel for
    for (long i = 0; i < coo.num_edges(); i += BLOCK_EDGES) {
        auto start = i;
        auto end = i + BLOCK_EDGES;
        if (end > coo.num_edges()) end = coo.num_edges();
        std::vector<int64_t> src, dst;
        std::tie(src, dst) = edge_accessor.slice(start, end);
        for (long i = 0; i < end - start; ++i) {
            auto src_id = src[i];
            auto dst_id = dst[i];
            long idx_pos;
            {
                std::lock_guard<std::mutex> lock(mutexes[src_id%64]);
                idx_pos = ptr_current[src_id]++;
            }
            idx_accessor.put(dst_id, idx_pos);
        }
    }
    for (int i = 0; i < num_nodes; ++i) {
        CHECK_EQ(ptr_current[i], ptr_tensor_[i+1].item<long>());
    }
    LOG(INFO) << "CSR complete";
    }

    // idx_store.save_to(TMPDIR + std::string("csr_idx"));
    return {std::move(ptr_store), std::move(idx_store)};

}


// CSRStore methods

CSRArrays CSRStore::tensor() const {
    return {ptr_store.tensor(), idx_store.tensor()};
}

torch::Tensor CSRStore::out_neighbors(long nid) {
    long start, end;
    // better dynamic dispatch?
    if (ptr_store.itemsize() == 4) {
        start = ptr_store.at(nid).item<int>();
        end = ptr_store.at(nid+1).item<int>();
    } else {
        start = ptr_store.at(nid).item<long>();
        end = ptr_store.at(nid+1).item<long>();
    }

    return idx_store.slice(start, end).tensor();
}

}


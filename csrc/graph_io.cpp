#include <vector>
#include <tuple>
#include <unordered_map>
#include <torch/torch.h>
#include "graph_io.hpp"

namespace gnnos {

// COOStore methods

COOArrays COOStore::tensor() const {
    return {src_store.tensor(), dst_store.tensor()};
}

COOStore COOStore::slice(long start, long end) const {
    return COOStore(
        this->src_store.slice(start, end),
        this->dst_store.slice(start, end),
        this->num_nodes());
}

std::tuple<long, TensorInfo, TensorInfo>
save_COOStore(const COOStore &coo, std::string path) {
    auto src_info = coo.src_store.save_to(path + ".coo_src");
    auto dst_info = coo.dst_store.save_to(path + ".coo_dst");
    return {coo.num_nodes(), src_info, dst_info};
}

// CSRStore methods

// TODO: use external sort instead of random fetching
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

    #pragma omp parallel for num_threads(IO_THREADS)
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
    // std::vector<std::mutex> mutexes(64);
    #pragma omp parallel for schedule(dynamic) num_threads(IO_THREADS)
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
            #pragma omp atomic capture
            idx_pos = ptr_current[src_id]++;
            // {
            //     std::lock_guard<std::mutex> lock(mutexes[src_id%64]);
            //     idx_pos = ptr_current[src_id]++;
            // }
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

CSRArrays CSRStore::tensor() const {
    return {ptr_store.tensor(), idx_store.tensor()};
}

torch::Tensor CSRStore::out_neighbors(long nid) {
    long start, end;
    // better dynamic dispatch?
    if (ptr_store.dtype() == torch::kInt) {
        start = ptr_store.at(nid).item<int>();
        end = ptr_store.at(nid+1).item<int>();
    } else {
        start = ptr_store.at(nid).item<long>();
        end = ptr_store.at(nid+1).item<long>();
    }

    return idx_store.slice(start, end).tensor();
}

std::tuple<long, TensorInfo, TensorInfo>
save_CSRStore(const CSRStore &csr, std::string path) {
    auto ptr_info = csr.ptr_store.save_to(path + ".csr_ptr");
    auto idx_info = csr.idx_store.save_to(path + ".idx_ptr");
    return {csr.num_nodes(), ptr_info, idx_info};
}

}

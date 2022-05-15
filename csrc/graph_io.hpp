#ifndef GNNOS_GRAPH_IO_HPP_
#define GNNOS_GRAPH_IO_HPP_

#include <vector>
#include <tuple>
#include <string>
#include <torch/torch.h>
#include "tensor_io.hpp"

namespace gnnos {

using COOArrays = std::tuple<torch::Tensor, torch::Tensor>;

class COOStore {
public:
    COOStore() = default;
    COOStore(const COOStore &) = default;
    COOStore(COOStore &&) = default;
    virtual ~COOStore() {}

    // initialize from two TensorStore's
    COOStore(TensorStore src_store, TensorStore dst_store, long num_nodes)
        : src_store_(std::move(src_store)), dst_store_(std::move(dst_store))
        , num_nodes_(num_nodes), num_edges_(src_store.numel())
    {
        CHECK_EQ(src_store_.shape().size(), 1) <<
            "COOStore: expect src/dst array to be flattened";
        CHECK_EQ(src_store_.shape(), dst_store_.shape()) <<
            "COOStore: expect src and dst arrays to have the same shape";
    }

    // initialize by splitting a contiguous tensor
    COOStore(TensorStore &combined, size_t num_nodes)
        : COOStore(combined.slice(0, combined.numel()/2),
                   combined.slice(combined.numel()/2, combined.numel()),
                   num_nodes)
    {}

    COOStore clone(std::string path, bool fill=true);
    COOStore slice(size_t start, size_t end) const;
    COOStore slice(size_t end) const { return this->slice(0, end); }

    long num_nodes() const { return num_nodes_; }
    long num_edges() const { return num_edges_; }
    std::tuple<size_t, TensorInfo, TensorInfo> metadata() const {
        return {num_nodes_, src_store_.metadata(), dst_store_.metadata()};
    }

    template <typename S, typename D=S>
    class Accessor {
    public:
        Accessor(const COOStore &coo)
        : src_accessor(coo.src_store_), dst_accessor(coo.dst_store_)
        {
            CHECK_EQ(src_accessor.size(), dst_accessor.size()) <<
                "COOStore::EdgeAccessor: expect src and dst to have the same size";
        }

        long size() const { return src_accessor.size(); }

        std::pair<S, D> operator[](size_t idx) const {
            return std::make_pair(src_accessor[idx], dst_accessor[idx]);
        }
        void put(const std::pair<S, D> &edge, size_t idx) {
            src_accessor.put(edge.first, idx);
            dst_accessor.put(edge.second, idx);
        }
        std::pair<std::vector<S>, std::vector<D>> slice(size_t start, size_t end) const {
            return std::make_pair(src_accessor.slice(start, end),
                                  dst_accessor.slice(start, end));
        }
        void slice_put(const S *src_data, const D *dst_data, size_t start, size_t end) {
            src_accessor.slice_put(src_data, start, end);
            dst_accessor.slice_put(dst_data, start, end);
        }
    private:
        TensorStore::Accessor<S> src_accessor;
        TensorStore::Accessor<D> dst_accessor;
    };

    template <typename S, typename D=S>
    Accessor<S, D> accessor() const& {
        return Accessor<S, D>(*this);
    }
    template <typename S, typename D=S>
    Accessor<S, D> accessor() && = delete;

protected:
    TensorStore src_store_, dst_store_;
    long num_nodes_, num_edges_;
};

struct CSRStore {
    CSRStore() = default;
    CSRStore(const CSRStore &) = default;
    CSRStore(CSRStore &&) = default;

    static CSRStore NewFrom(const COOStore &);

    long num_nodes() const { return ptr_store.numel() - 1; }
    long num_edges() const { return idx_store.numel(); }

    TensorStore ptr_store;
    TensorStore idx_store;
};


} // namespace gnnos

#endif

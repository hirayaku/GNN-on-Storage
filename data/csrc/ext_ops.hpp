#pragma once
#include <vector>
#include <tuple>
#include <torch/extension.h>

using Tensor = torch::Tensor;
using EdgeType = std::tuple<Tensor, Tensor>;

// get the destination index of the scatter-append operation
Tensor &scatter_index(Tensor &out, const Tensor &input, const Tensor &intervals);

// copy src to out based on the scatter index `index`, dim = 0
Tensor &scatter_copy(Tensor &out, const Tensor &index, const Tensor &src);

// gather intervals from src tensor to out tensor
Tensor &ranges_gather(Tensor &out, const Tensor &src, const Tensor &, const Tensor &);
Tensor &ranges_add(Tensor &target, const Tensor &, const Tensor &, const Tensor &);

// merge the input list of edges in COO formats (assuming edges are sorted by dst)
// doesn't relabel nodes; just merge the COOs into a (n,n) adj matrix
// @return tuple of <colptr, row>
std::tuple<Tensor, Tensor> coo_list_merge(
    long num_nodes, const std::vector<EdgeType> &edges
);

// @return tuple of <colptr, row>
std::tuple<Tensor, Tensor> coo_ranges_merge(
    long num_nodes,
    const std::vector<EdgeType> &coo_list,
    const std::vector<Tensor> &starts,
    const std::vector<Tensor> &sizes
);

// optional: the following is for better sequential IO performance with mmap
bool check_madv_populate();

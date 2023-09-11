#pragma once
#include <cmath>
#include <torch/torch.h>
#include "utils.hpp"
#include "packed.hpp"

namespace fennel {

template <typename T>
inline auto adj(int64_t *ptr, T *data, int64_t v)
{
    return Slice<T>(data + ptr[v], data + ptr[v + 1]);
}

inline float balance_score(int64_t size, float gamma, float alpha) {
    return -alpha * gamma * std::pow(size, gamma-1);
}

using Tensor = torch::Tensor;
using OptTensor = torch::optional<torch::Tensor>;

/**
 * A basic implementation of Fennel
 * 
 * Accepts an optional `init_partition` parameter to help with the implementation of reFennel
 * reFennel is not done here in C++, but in Python for convenience
*/
Tensor partition(
    Tensor ptr, Tensor idx, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition = torch::nullopt
);

/**
 * Optimized version of Fennel; complexity reduced from O(m + n*k) to O(m + n*log(k))
 * Based on the observation that when deciding the partitioning of node v,
 * we only need to consider at most connected_p(v) <= deg(v) partitions
 * 
 * Accepts an extra `scan_thres` parameter to control when to switch from
 * connected_p(v) random accesses to k sequential accesses.
 * If connected_p(v) > k * scan_thres, we check all k partition in sequential instead
 **/
Tensor partition_opt(
    Tensor ptr, Tensor idx, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
);

/**
 * Parallelized version of the vanilla fennel; only good for a small k.
 * Per-partition scores could be stale but have eventual consistency
*/
Tensor partition_par(
    Tensor ptr, Tensor idx, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
);


/**
 * Multi-strata fennel
 * 
 * The element in `labels` should range from 0, 1, ..., NL-1
 * If `labels` is not None, it balances the label distribution in each partition
 * The `alphas` parameter decides the importance of balancing each label.
 * If alphas[L] = m * pow(k, gamma-1) / pow(n_L, gamma), where n_L is #nodes with label L,
 * each label is treated with equal importance, and the label balance is treated as equally
 * important as minimizing edge cuts.
*/
Tensor partition_stratified(
    Tensor ptr, Tensor idx, int64_t k, OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack, double label_slack,
    OptTensor init_partition = torch::nullopt
);
Tensor partition_stratified_opt(
    Tensor ptr, Tensor idx, int64_t k, OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack, double label_slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
);


/**
 * Fennel with weighted edges
*/
Tensor partition_weighted(
    Tensor ptr, Tensor idx, OptTensor weights, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres = torch::nullopt
);

Tensor partition_stratified_weighted(
    Tensor ptr, Tensor idx, OptTensor weights, int64_t k, OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack, double label_slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres = torch::nullopt
);

}
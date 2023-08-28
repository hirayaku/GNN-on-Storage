#include <cmath>
#include <ctime>
#include <vector>
#include <map>
#include <functional>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <atomic>
#include <stdlib.h>
#include <omp.h>
#include <torch/extension.h>
#include "utils.hpp"
#include "packed.hpp"

inline auto adj(int64_t *ptr, int64_t *idx, int64_t v)
{
    return Slice<int64_t>(idx + ptr[v], idx + ptr[v + 1]);
}

inline float balance_score(int64_t size, float gamma, float alpha) {
    return -alpha * gamma * std::pow(size, gamma-1);
}

using Tensor = torch::Tensor;
using OptTensor = torch::optional<Tensor>;

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
) {
    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    int64_t cap = (int64_t)((float)n / k * slack);
    float alphaf = alpha, gammaf = gamma;
    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");

    // prepare map: node->partition and map: partition->size
    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once");
    Tensor init_ptn_tensor = init_partition.value_or(torch::ones(n, torch::kInt32)*k);
    Tensor ptn_sizes = init_ptn_tensor.to(torch::kFloat).histc(k+1, 0, k+1).to(torch::kInt64);
    auto node2ptn = Slice<int>::from_tensor(init_ptn_tensor).to_vec();
    TORCH_CHECK(node2ptn.size() == n);
    auto ptn_nodes = Slice<int64_t>::from_tensor(ptn_sizes).to_vec();
    TORCH_CHECK(ptn_nodes.size() == k+1);

    // compute initial balance scores
    std::vector<float> ptn_balance_scores(k, 0);
    std::transform(
        ptn_nodes.begin(), ptn_nodes.begin()+k, ptn_balance_scores.begin(),
        [=](int64_t ptn_size) { return balance_score(ptn_size, gammaf, alphaf); }
    );

    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    const float min_score = balance_score(n, gammaf, alphaf);
    srand(time(nullptr));
    for (auto v : node_stream)
    {
        // random assign a partition if no partition is found
        int v_ptn = rand() % k;
        // remove v from its partition
        // O(1)
        int old_ptn = node2ptn[v];
        if (old_ptn != k)
        {
            // XXX update global state
            ptn_balance_scores[old_ptn] = balance_score(ptn_nodes[old_ptn]-1, gammaf, alphaf);
            --ptn_nodes[old_ptn];
        }
        // O(deg(v))
        auto v_adj = adj(ptr_data, idx_data, v);
        for (int64_t ngh : v_adj)
        {
            int ptn = node2ptn[ngh];
            ++ptn_neighbors[ptn];
        }
        // get the partition having the maximum score
        // O(k)
        float current_max = min_score;
        for (size_t i = 0; i < k; ++i)
        {
            if (ptn_nodes[i] >= cap)
                continue;
            float ptn_score = ptn_neighbors[i] + ptn_balance_scores[i];
            if (ptn_score > current_max)
            {
                current_max = ptn_score;
                v_ptn = i;
            }
        }
        // move v from old_ptn to v_ptn
        // O(1)
        node2ptn[v] = v_ptn;
        // XXX update global state
        ptn_balance_scores[v_ptn] = balance_score(ptn_nodes[v_ptn]+1, gammaf, alphaf);
        ++ptn_nodes[v_ptn];
        // cleaning up
        // O(k)
        std::fill(ptn_neighbors.begin(), ptn_neighbors.end(), 0);
    }

    return vector_to_tensor(node2ptn);
}

template <typename MapT, typename RMapT>
void check_consistency(
    const MapT &map,
    const RMapT &reverse_map,
    std::string msg
) {
    TORCH_CHECK(map.size() == reverse_map.size(),
        msg, map.size(), "!=", reverse_map.size());
    for (auto rev_pair : reverse_map) {
        auto val = rev_pair.first;
        auto key = rev_pair.second;
        TORCH_CHECK(map[key] == val, msg, "Inconsistent value for key ", key);
    }
}

/**
 * Optimized version of Fennel; complexity reduced from O(m + n*k) to O(m + n*log(k)})
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
) {
    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    int64_t cap = (int64_t)((float)n / k * slack);
    float alphaf = alpha, gammaf = gamma;
    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");

    // prepare map: node->partition and map: partition->size
    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once");
    Tensor init_ptn_tensor = init_partition.value_or(torch::ones(n, torch::kInt32)*k);
    Tensor ptn_sizes = init_ptn_tensor.to(torch::kFloat).histc(k+1, 0, k+1).to(torch::kInt64);
    TORCH_CHECK(ptn_sizes.sum().item().to<int64_t>() == n);
    auto node2ptn = Slice<int>::from_tensor(init_ptn_tensor).to_vec();
    TORCH_CHECK(node2ptn.size() == n);
    auto ptn_nodes = Slice<int64_t>::from_tensor(ptn_sizes).to_vec();
    TORCH_CHECK(ptn_nodes.size() == k+1);

    // map: partition -> balance_score and ord-map: balance_score -> partition
    // the states of these two maps should be **consistent**
    std::vector<float> ptn_balance_scores(k, 0);
    auto cmp_score = [&](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
        // we need to break ties when scores are equal
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
    };
    std::set<std::pair<float, int>, decltype(cmp_score)> ord_scores(cmp_score);
    // compute initial balance scores
    std::transform(
        ptn_nodes.begin(), ptn_nodes.begin()+k, ptn_balance_scores.begin(),
        [=](int64_t ptn_size) { return balance_score(ptn_size, gammaf, alphaf); }
    );
    for (int i = 0; i < k; ++i) {
        ord_scores.insert({ptn_balance_scores[i], i});
    }
    #ifndef NDEBUG
    check_consistency(ptn_balance_scores, ord_scores, "[init]");
    #endif

    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    std::vector<int> ptn_to_check; ptn_to_check.reserve(k);
    const float min_score = balance_score(n, gammaf, alphaf);
    srand(time(nullptr));
    for (auto v : node_stream)
    {
        // random assign a partition if no partition is found
        int v_ptn = rand() % k;
        // remove v from its partition
        // O(log(k))
        int old_ptn = node2ptn[v];
        if (old_ptn != k)
        {
            // XXX update global state
            float old_ptn_score = ptn_balance_scores[old_ptn];
            float updated_score = balance_score(ptn_nodes[old_ptn]-1, gammaf, alphaf);
            // replace {old_score, ptn} with {new_score, ptn}
            size_t erased = ord_scores.erase({old_ptn_score, old_ptn});
            TORCH_CHECK(
                erased == 1,
                "remove v: should erase exactly 1 element but see ", erased
            );
            ord_scores.insert({updated_score, old_ptn});
            ptn_balance_scores[old_ptn] = updated_score;
            --ptn_nodes[old_ptn];
        }
        // O(deg(v))
        auto v_adj = adj(ptr_data, idx_data, v);
        for (int64_t ngh : v_adj)
        {
            int ptn = node2ptn[ngh];
            if (ptn != k && ptn_neighbors[ptn] == 0) ptn_to_check.push_back(ptn);
            ++ptn_neighbors[ptn];
        }
        // get the partition having the maximum score
        // O(deg(v))
        float current_max = min_score;
        if (ptn_to_check.size() >= k * scan_thres.value_or(1.0)) {
            for (size_t i = 0; i < k; ++i)
            {
                if (ptn_nodes[i] >= cap) continue;
                float ptn_score = ptn_neighbors[i] + ptn_balance_scores[i];
                if (ptn_score > current_max)
                {
                    current_max = ptn_score;
                    v_ptn = i;
                }
            }
        } else {
            // only check connected partitions
            for (auto p : ptn_to_check) {
                if (ptn_nodes[p] >= cap) continue;
                float ptn_score = ptn_neighbors[p] + ptn_balance_scores[p];
                if (ptn_score > current_max) {
                    current_max = ptn_score;
                    v_ptn = p;
                }
            }
            auto score_pair = *ord_scores.begin();
            // an untouched partition has larger score than current_max
            if (v_ptn != score_pair.second && score_pair.first > current_max) {
                v_ptn = score_pair.second;
            }
        }

        // assign v to v_ptn
        // O(log(k))
        node2ptn[v] = v_ptn;
        // XXX update global state
        float old_ptn_score = ptn_balance_scores[v_ptn];
        float updated_score = balance_score(ptn_nodes[v_ptn]+1, gammaf, alphaf);
        size_t erased = ord_scores.erase({old_ptn_score, v_ptn});
        TORCH_CHECK(
            erased == 1,
            "add v: should erase exactly 1 element but see ", erased
        );
        ord_scores.insert({updated_score, v_ptn});
        ptn_balance_scores[v_ptn] = updated_score;
        ++ptn_nodes[v_ptn];
        // cleaning up
        // O(deg(v))
        if (ptn_to_check.size() > k / scan_thres.value_or(1.0)) {
            std::fill(ptn_neighbors.begin(), ptn_neighbors.end(), 0);
        } else {
            for (auto p : ptn_to_check) ptn_neighbors[p] = 0;
        }
        ptn_to_check.clear();

        #ifndef NDEBUG
        std::ostringstream oss;
        oss << "[v=" << v << "]";
        check_consistency(ptn_balance_scores, ord_scores, oss.str());
        #endif
    }

    return vector_to_tensor(node2ptn);
}

// // per-partition data packing size and score: [size: int, score: score]
// using packed_t = int64_t;
// // need better tests for unpack & pack
// inline static int32_t extract_size(packed_t packed) {
//     uint32_t upper_bits = (uint64_t)packed >> 32;
//     return *(int32_t *)&upper_bits;
// }
// inline static float extract_score(packed_t packed) {
//     uint32_t lower_bits = ((uint64_t)packed << 32) >> 32;
//     return *(float *)(&lower_bits);
// }
// inline static std::tuple<int32_t, float> unpack(packed_t packed) {
//     return {extract_size(packed), extract_score(packed)};
// }
// inline static packed_t pack(int32_t size, float score) {
//     packed_t packed = ((packed_t) size) << 32;
//     uint32_t score_bits = *(uint32_t *)&score;
//     return packed | (packed_t)score_bits;
// }
/**
 * Parallelized version of the vanilla fennel; only good for a small k.
 * Per-partition scores could be stale but have eventual consistency
*/
Tensor partition_par(
    Tensor ptr, Tensor idx, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
) {
    int num_par = omp_thread_count();
    srand(time(nullptr));
    // per-thread RNG state
    std::vector<unsigned> rand_states(num_par, 0);
    std::generate(rand_states.begin(), rand_states.end(), rand);

    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    int64_t cap = (int64_t)((float)n / k * slack);
    float alphaf = alpha, gammaf = gamma;
    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");

    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once");
    Tensor init_ptn_tensor = init_partition.value_or(torch::ones(n, torch::kInt32)*k);
    Tensor ptn_sizes = init_ptn_tensor.to(torch::kFloat).histc(k+1, 0, k+1).to(torch::kInt64);
    // map: node->partition, techniquely it should be a container of atomics
    auto node2ptn = Slice<int>::from_tensor(init_ptn_tensor).to_vec();
    TORCH_CHECK(node2ptn.size() == n);

    auto update = [=](packed_t pdata, int size_delta) {
        auto new_size = extract_int(pdata) + size_delta;
        auto new_score = balance_score(new_size, gammaf, alphaf);
        return pack(new_size, new_score);
    };
    // initial per-partition data
    std::vector<std::atomic<packed_t>> ptn2data(k);
    auto sizes = Slice<int64_t>::from_tensor(ptn_sizes).to_vec();
    for (int i = 0; i < k; ++i) {
        float score = balance_score(sizes[i], gammaf, alphaf);
        auto pdata = pack(sizes[i], score);
        ptn2data[i].store(pdata);
    }

    const float min_score = balance_score(n, gammaf, alphaf);
    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    auto stream_it = node_stream.begin();
    #pragma omp parallel
    {
    std::vector<int> ptn_neighbors(k + 1, 0);
    #pragma omp for schedule(dynamic)
    // for (auto v : node_stream) {
    for (auto it = stream_it; it < node_stream.end(); ++it) {
        int pid = omp_get_thread_num();
        auto v = *it;
        // random assign a partition if no partition is found
        int v_ptn = rand_r(&rand_states[pid]) % k;
        // remove v from its partition
        // O(1)
        int old_ptn = node2ptn[v];
        if (old_ptn != k)
        {
            auto pdata = ptn2data[old_ptn].load(std::memory_order_relaxed);
            while (!ptn2data[old_ptn].compare_exchange_weak(pdata, update(pdata, -1), std::memory_order_relaxed));
        }
        // O(deg(v))
        auto v_adj = adj(ptr_data, idx_data, v);
        for (int64_t ngh : v_adj)
        {
            int ptn = node2ptn[ngh];
            ++ptn_neighbors[ptn];
        }
        // get the partition having the maximum score
        // O(k)
        float current_max = min_score;
        for (size_t i = 0; i < k; ++i)
        {
            auto tuple = unpack(ptn2data[i].load());
            if (std::get<0>(tuple) >= cap) continue;
            float ptn_score = ptn_neighbors[i] + std::get<1>(tuple);
            if (ptn_score > current_max)
            {
                current_max = ptn_score;
                v_ptn = i;
            }
        }
        // move v from old_ptn to v_ptn
        // O(1)
        node2ptn[v] = v_ptn;
        auto pdata = ptn2data[v_ptn].load(std::memory_order_relaxed);
        while (!ptn2data[v_ptn].compare_exchange_weak(pdata, update(pdata, 1), std::memory_order_relaxed));
        // cleaning up
        // O(k)
        std::fill(ptn_neighbors.begin(), ptn_neighbors.end(), 0);
    }
    }

    return vector_to_tensor(node2ptn);
}


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
    Tensor ptr, Tensor idx, int64_t k,
    OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack,
    OptTensor init_partition = torch::nullopt
) {
    if (labels == torch::nullopt) {
        float alpha = alphas.index({0}).item().to<float>();
        return partition_opt(
            ptr, idx, k, node_order, gamma, alpha, slack, init_partition
        );
    }

    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    // map: label -> cap
    auto label_tensor = labels.value();
    int NL = label_tensor.max().item().to<int>() + 1;
    const std::vector<float> label2cap = tensor_to_vector<float>(
        label_tensor.to(torch::kFloat).histc(NL, 0, NL) / k * slack
    );
    // map: node -> label
    const std::vector<int> node2label = tensor_to_vector<int>(
        label_tensor.to(torch::kInt32)
    );
    // #ifndef NDEBUG
    // std::cout << '[' << std::fixed << std::setprecision(2);
    // std::copy(label2cap.begin(), label2cap.end(), std::ostream_iterator<float>(std::cout, ", "));
    // std::cout << ']' << std::setprecision(-1) << std::endl;
    // #endif

    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");
    
    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once"
        );
    // map: node -> partition
    auto node2ptn = tensor_to_vector<int>(
        init_partition.value_or(torch::ones(n, torch::kInt32)*k)
    );
    TORCH_CHECK(node2ptn.size() == n);
    // map: label -> alpha
    std::vector<float> label2alpha = tensor_to_vector<float>(alphas);
    // map: label -> (map: ptn -> size)
    std::vector<std::vector<int64_t>> label2sizes(NL, std::vector<int64_t>(k+1, 0));
    for (int64_t i = 0; i < n; ++i) {
        int label = node2label[i];
        if (label >= 0) {
            int ptn = node2ptn[i];
            label2sizes[label][ptn]++;
        }
    }

    // map: label -> (map: ptn -> score)
    // compute initial balance scores
    std::vector<std::vector<float>> label2scores(NL, std::vector<float>(k, 0));
    for (int label = 0; label < NL; ++label) {
        float alpha = label2alpha[label];
        std::transform(
            label2sizes[label].begin(), label2sizes[label].begin()+k, label2scores[label].begin(),
            [=](int64_t ptn_size) { return balance_score(ptn_size, gamma, alpha); }
        );
    }

    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    const float min_score = balance_score(n, gamma, alphas.max().item().to<float>());
    srand(time(nullptr));
    for (auto v : node_stream)
    {
        int v_ptn = rand() % k;
        auto v_adj = adj(ptr_data, idx_data, v);
        int label = node2label[v];
        if (label < 0) {
            // for unbalanced nodes, select the partition with largest
            // O(deg(v))
            for (int64_t ngh : v_adj)
            {
                int ptn = node2ptn[ngh];
                ++ptn_neighbors[ptn];
            }
            float current_max = 0;
            for (size_t i = 0; i < k; ++i)
            {
                if (ptn_neighbors[i] > current_max)
                {
                    current_max = ptn_neighbors[i];
                    v_ptn = i;
                }
            }
        } else {
            std::vector<float> &ptn_balance_scores = label2scores[label];
            std::vector<int64_t> &ptn_nodes = label2sizes[label];
            float alpha = label2alpha[label];
            float cap = label2cap[label];
            // remove v from its original partition
            // O(1)
            int old_ptn = node2ptn[v];
            if (old_ptn != k)
            {
                ptn_balance_scores[old_ptn] = balance_score(ptn_nodes[old_ptn]-1, gamma, alpha);
                --ptn_nodes[old_ptn];
            }
            // O(deg(v))
            for (int64_t ngh : v_adj)
            {
                int ptn = node2ptn[ngh];
                ++ptn_neighbors[ptn];
            }
            // get the partition having the maximum score
            // O(k)
            float current_max = min_score;
            for (size_t i = 0; i < k; ++i)
            {
                if (ptn_nodes[i] >= cap)
                    continue;
                float ptn_score = ptn_neighbors[i] + ptn_balance_scores[i];
                if (ptn_score > current_max)
                {
                    current_max = ptn_score;
                    v_ptn = i;
                }
            }
            // add v to the decided partition
            ptn_balance_scores[v_ptn] = balance_score(ptn_nodes[v_ptn]+1, gamma, alpha);
            ++ptn_nodes[v_ptn];
        }
        // O(1)
        node2ptn[v] = v_ptn;
        // cleaning up
        // O(k)
        std::fill(ptn_neighbors.begin(), ptn_neighbors.end(), 0);
    }

    return vector_to_tensor(node2ptn);
}

Tensor partition_stratified_opt(
    Tensor ptr, Tensor idx, int64_t k,
    OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
) {
    if (labels == torch::nullopt) {
        float alpha = alphas.index({0}).item().to<float>();
        return partition_par(
            ptr, idx, k, node_order, gamma, alpha, slack, init_partition
        );
    }

    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    // map: label -> cap
    auto label_tensor = labels.value();
    int NL = label_tensor.max().item().to<int>() + 1;
    // auto label_hist = label_tensor.to(torch::kDouble).histc(NL, 0, NL);
    // double hist_sum = label_hist.sum().item<double>();
    // TORCH_CHECK(hist_sum == n, "#labels = ", NL,
    //     ", labeled node count = ", hist_sum, " != ", n);
    const std::vector<double> label2cap = tensor_to_vector<double>(
        label_tensor.to(torch::kDouble).histc(NL, 0, NL) / k * slack
    );
    // map: node -> label
    const std::vector<int> node2label = tensor_to_vector<int>(
        label_tensor.to(torch::kInt32)
    );

    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");
    
    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once"
        );
    // map: node -> partition
    auto node2ptn = tensor_to_vector<int>(
        init_partition.value_or(torch::ones(n, torch::kInt32)*k)
    );
    TORCH_CHECK(node2ptn.size() == n);
    // map: label -> alpha
    std::vector<float> label2alpha = tensor_to_vector<float>(alphas);
    // map: label -> (map: ptn -> size)
    std::vector<std::vector<int64_t>> label2sizes(NL, std::vector<int64_t>(k+1, 0));
    for (int64_t i = 0; i < n; ++i) {
        int label = node2label[i];
        if (label >= 0) {
            int ptn = node2ptn[i];
            label2sizes[label][ptn]++;
        }
    }

    // map: label -> (map: ptn -> score)
    // compute initial balance scores
    std::vector<std::vector<float>> label2scores(NL, std::vector<float>(k, 0));
    for (int label = 0; label < NL; ++label) {
        float alpha = label2alpha[label];
        std::transform(
            label2sizes[label].begin(), label2sizes[label].begin()+k, label2scores[label].begin(),
            [=](int64_t ptn_size) {
                return balance_score(ptn_size, gamma, alpha);
                // double score = balance_score(ptn_size, gamma, alpha);
                // std::cout << "label=" << label << " ptn_size=" << ptn_size
                //     << " gamma=" << gamma << " alpha=" << alpha
                //     << " score=" << score << "\n";
                // return score;
            }
        );
    }
    // map: label -> (balance_score -> partition)
    auto cmp_score = [&](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
        // we need to break ties when scores are equal
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
    };
    using rev_scoreset = std::set<std::pair<float, int>, decltype(cmp_score)>;
    std::vector<rev_scoreset> label2ord_scores(NL, rev_scoreset(cmp_score));
    for (int label = 0; label < NL; ++label) {
        auto &scores = label2scores[label];
        auto &ord_scores = label2ord_scores[label];
        for (int i = 0; i < k; ++i) {
            ord_scores.insert({scores[i], i});
        }
        #ifndef NDEBUG
        std::ostringstream oss;
        oss << "[init, label=" << label << "]";
        check_consistency(scores, ord_scores, oss.str());
        #endif
    }

    const float min_score = balance_score(n, gamma, alphas.max().item().to<float>());
    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    std::vector<int> ptn_to_check; ptn_to_check.reserve(k);
    for (auto v : node_stream)
    {
        // random assign a partition if no partition is found
        int v_ptn = rand() % k;
        auto v_adj = adj(ptr_data, idx_data, v);
        int label = node2label[v];
        // probe neighbors: O(deg(v))
        for (int64_t ngh : v_adj)
        {
            int ptn = node2ptn[ngh];
            // if (ptn_neighbors[ptn] == 0) ptn_to_check.push_back(ptn);
            if (ptn != k && ptn_neighbors[ptn] == 0) ptn_to_check.push_back(ptn);
            ++ptn_neighbors[ptn];
        }

        if (label < 0) {
            // for nodes no need to balance, select the closest partition
            float current_max = 0;
            for (auto p : ptn_to_check) {
                if (ptn_neighbors[p] > current_max) {
                    current_max = ptn_neighbors[p];
                    v_ptn = p;
                }
            }
            node2ptn[v] = v_ptn;
        } else {
            // balance score per partition for current label
            std::vector<float> &ptn_balance_scores = label2scores[label];
            // size per partition for current label
            std::vector<int64_t> &ptn_nodes = label2sizes[label];
            // ordered score per partition for current label
            rev_scoreset &ord_scores = label2ord_scores[label];
            // alpha for the current label
            float alpha = label2alpha[label];
            // cap for the current label
            float cap = label2cap[label];
            // remove v from its original partition
            // O(log(k))
            int old_ptn = node2ptn[v];

            if (old_ptn != k) {
                // XXX writes
                float old_ptn_score = ptn_balance_scores[old_ptn];
                float updated_score = balance_score(ptn_nodes[old_ptn]-1, gamma, alpha);
                // replace {old_score, ptn} with {new_score, ptn}
                size_t erased = ord_scores.erase({old_ptn_score, old_ptn});
                TORCH_CHECK(
                    erased == 1,
                    "remove v: should erase exactly 1 element but see ", erased
                );
                ord_scores.insert({updated_score, old_ptn});
                ptn_balance_scores[old_ptn] = updated_score;
                --ptn_nodes[old_ptn];
            }

            // get the partition having the maximum score
            // O(deg(v)+log(k))
            float current_max = min_score;
            if (ptn_to_check.size() >= k * scan_thres.value_or(1.0)) {
                for (size_t i = 0; i < k; ++i)
                {
                    if (ptn_nodes[i] >= cap) continue;
                    float ptn_score = ptn_neighbors[i] + ptn_balance_scores[i];
                    if (ptn_score > current_max)
                    {
                        current_max = ptn_score;
                        v_ptn = i;
                    }
                }
            } else {
                // only check connected partitions
                for (auto p : ptn_to_check) {
                    if (ptn_nodes[p] >= cap) continue;
                    float ptn_score = ptn_neighbors[p] + ptn_balance_scores[p];
                    if (ptn_score > current_max) {
                        current_max = ptn_score;
                        v_ptn = p;
                    }
                }
                // an untouched partition has larger score than current_max
                auto score_pair = *ord_scores.begin();
                if (v_ptn != score_pair.second && score_pair.first > current_max) {
                    v_ptn = score_pair.second;
                }
                TORCH_CHECK(ptn_nodes[v_ptn] <= cap, "the assigned partition exceeds the label cap");
            }
            // add v to partition v_ptn
            node2ptn[v] = v_ptn;
            TORCH_CHECK(v_ptn != k);

            // protect thread-unsafe std::set
            float old_ptn_score = ptn_balance_scores[v_ptn];
            float updated_score = balance_score(ptn_nodes[v_ptn]+1, gamma, alpha);
            // XXX: writes
            ptn_balance_scores[v_ptn] = updated_score;
            size_t erased = ord_scores.erase({old_ptn_score, v_ptn});
            TORCH_CHECK(
                erased == 1,
                "add v from ", old_ptn, "->", v_ptn,
                ": should erase exactly 1 element but see ", erased
            );
            ord_scores.insert({updated_score, v_ptn});
            ++ptn_nodes[v_ptn];

            #ifndef NDEBUG
            std::ostringstream oss;
            oss << "[v=" << v << "]";
            check_consistency(ptn_balance_scores, ord_scores, oss.str());
            #endif
        } // labeled case ends
        // cleaning up
        // O(deg(v))
        for (auto p : ptn_to_check) ptn_neighbors[p] = 0;
        ptn_to_check.clear();
    }

    // // dump label 
    // for (int label = 0; label < NL; ++label) {
    //     std::vector<int64_t> &ptn_nodes = label2sizes[label];
    //     auto iter = std::max_element(ptn_nodes.begin(), ptn_nodes.end());
    //     std::cout << "Label " << label << ": max-element = " << *iter
    //         << ", Ptn " << iter - ptn_nodes.begin()
    //         << "(cap = " << label2cap[label] << ")\n";
    // }

    return vector_to_tensor(node2ptn);
}

// not scaling with locks
Tensor partition_stratified_par(
    Tensor ptr, Tensor idx, int64_t k,
    OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack,
    OptTensor init_partition = torch::nullopt,
    torch::optional<double> scan_thres=torch::nullopt
) {
    if (labels == torch::nullopt) {
        float alpha = alphas.index({0}).item().to<float>();
        return partition_par(
            ptr, idx, k, node_order, gamma, alpha, slack, init_partition
        );
    }

    int num_par = omp_thread_count();
    srand(time(nullptr));
    // per-thread RNG state
    std::vector<unsigned> rand_states(num_par, 0);
    std::generate(rand_states.begin(), rand_states.end(), rand);

    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();

    // map: label -> cap
    auto label_tensor = labels.value();
    int NL = label_tensor.max().item().to<int>() + 1;
    const std::vector<float> label2cap = tensor_to_vector<float>(
        label_tensor.to(torch::kFloat).histc(NL, 0, NL) / k * slack
    );
    // map: node -> label
    const std::vector<int> node2label = tensor_to_vector<int>(
        label_tensor.to(torch::kInt32)
    );
    // #ifndef NDEBUG
    // std::cout << '[' << std::fixed << std::setprecision(2);
    // std::copy(label2cap.begin(), label2cap.end(), std::ostream_iterator<float>(std::cout, ", "));
    // std::cout << ']' << std::setprecision(-1) << std::endl;
    // #endif

    // node streaming order
    Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");
    
    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once"
        );
    // map: node -> partition
    auto node2ptn = tensor_to_vector<int>(
        init_partition.value_or(torch::ones(n, torch::kInt32)*k)
    );
    TORCH_CHECK(node2ptn.size() == n);
    // map: label -> alpha
    std::vector<float> label2alpha = tensor_to_vector<float>(alphas);
    // map: label -> (map: ptn -> size)
    std::vector<std::vector<int64_t>> label2sizes(NL, std::vector<int64_t>(k+1, 0));
    for (int64_t i = 0; i < n; ++i) {
        int label = node2label[i];
        if (label >= 0) {
            int ptn = node2ptn[i];
            label2sizes[label][ptn]++;
        }
    }

    // map: label -> (map: ptn -> score)
    // compute initial balance scores
    std::vector<std::vector<float>> label2scores(NL, std::vector<float>(k, 0));
    for (int label = 0; label < NL; ++label) {
        float alpha = label2alpha[label];
        std::transform(
            label2sizes[label].begin(), label2sizes[label].begin()+k, label2scores[label].begin(),
            [=](int64_t ptn_size) {
                return balance_score(ptn_size, gamma, alpha);
                // double score = balance_score(ptn_size, gamma, alpha);
                // std::cout << "label=" << label << " ptn_size=" << ptn_size
                //     << " score=" << score << "\n";
                // return score;
            }
        );
    }
    // map: label -> (balance_score -> partition)
    auto cmp_score = [&](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
        // we need to break ties when scores are equal
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
    };
    using rev_scoreset = std::set<std::pair<float, int>, decltype(cmp_score)>;
    std::vector<rev_scoreset> label2ord_scores(NL, rev_scoreset(cmp_score));
    for (int label = 0; label < NL; ++label) {
        auto &scores = label2scores[label];
        auto &ord_scores = label2ord_scores[label];
        for (int i = 0; i < k; ++i) {
            ord_scores.insert({scores[i], i});
        }
        #ifndef NDEBUG
        std::ostringstream oss;
        oss << "[init, label=" << label << "]";
        check_consistency(scores, ord_scores, oss.str());
        #endif
    }

    const float min_score = balance_score(n, gamma, alphas.max().item().to<float>());
    std::vector<omp_lock_t> locks(NL*16); // add extra padding to avoid cache aliasing
    #pragma omp parallel
    {
    // scratchpad vectors
    std::vector<int> ptn_neighbors(k + 1, 0);
    std::vector<int> ptn_to_check; ptn_to_check.reserve(k);
    #pragma omp for
    for (auto v : node_stream)
    {
        int pid = omp_get_thread_num();
        // random assign a partition if no partition is found
        int v_ptn = rand_r(&rand_states[pid]) % k;

        auto v_adj = adj(ptr_data, idx_data, v);
        int label = node2label[v];
        // probe neighbors: O(deg(v))
        for (int64_t ngh : v_adj)
        {
            int ptn = node2ptn[ngh];
            // if (ptn_neighbors[ptn] == 0) ptn_to_check.push_back(ptn);
            if (ptn != k && ptn_neighbors[ptn] == 0) ptn_to_check.push_back(ptn);
            ++ptn_neighbors[ptn];
        }

        if (label < 0) {
            // for nodes no need to balance, select the closest partition
            float current_max = 0;
            for (auto p : ptn_to_check) {
                if (ptn_neighbors[p] > current_max) {
                    current_max = ptn_neighbors[p];
                    v_ptn = p;
                }
            }
            node2ptn[v] = v_ptn;
        } else {
            // balance score per partition for current label
            std::vector<float> &ptn_balance_scores = label2scores[label];
            // size per partition for current label
            std::vector<int64_t> &ptn_nodes = label2sizes[label];
            // ordered score per partition for current label
            rev_scoreset &ord_scores = label2ord_scores[label];
            // alpha for the current label
            float alpha = label2alpha[label];
            // cap for the current label
            float cap = label2cap[label];
            // remove v from its original partition
            // O(log(k))
            int old_ptn = node2ptn[v];

            std::pair<float, int> score_pair;
            if (old_ptn != k || ptn_to_check.size() < k * scan_thres.value_or(1.0)) {
                // protect thread-unsafe std::set
                omp_set_lock(&locks[label*16]);
                if (old_ptn != k) {
                    // XXX writes
                    float old_ptn_score = ptn_balance_scores[old_ptn];
                    float updated_score = balance_score(ptn_nodes[old_ptn]-1, gamma, alpha);
                    // replace {old_score, ptn} with {new_score, ptn}
                    size_t erased = ord_scores.erase({old_ptn_score, old_ptn});
                    TORCH_CHECK(
                        erased == 1,
                        "remove v: should erase exactly 1 element but see ", erased
                    );
                    ord_scores.insert({updated_score, old_ptn});
                    ptn_balance_scores[old_ptn] = updated_score;
                    --ptn_nodes[old_ptn];
                }
                if (ptn_to_check.size() < k * scan_thres.value_or(1.0))
                    score_pair = *ord_scores.begin();
                omp_unset_lock(&locks[label*16]);
            }

            // get the partition having the maximum score
            // O(deg(v)+log(k))
            float current_max = min_score;
            if (ptn_to_check.size() >= k * scan_thres.value_or(1.0)) {
                for (size_t i = 0; i < k; ++i)
                {
                    if (ptn_nodes[i] >= cap) continue;
                    float ptn_score = ptn_neighbors[i] + ptn_balance_scores[i];
                    if (ptn_score > current_max)
                    {
                        current_max = ptn_score;
                        v_ptn = i;
                    }
                }
            } else {
                // only check connected partitions
                for (auto p : ptn_to_check) {
                    if (ptn_nodes[p] >= cap) continue;
                    float ptn_score = ptn_neighbors[p] + ptn_balance_scores[p];
                    if (ptn_score > current_max) {
                        current_max = ptn_score;
                        v_ptn = p;
                    }
                }
                // an untouched partition has larger score than current_max
                if (v_ptn != score_pair.second && score_pair.first > current_max) {
                    v_ptn = score_pair.second;
                }
            }
            // add v to partition v_ptn
            node2ptn[v] = v_ptn;

            // protect thread-unsafe std::set
            omp_set_lock(&locks[label*16]);
            float old_ptn_score = ptn_balance_scores[v_ptn];
            float updated_score = balance_score(ptn_nodes[v_ptn]+1, gamma, alpha);
            // XXX: writes
            ptn_balance_scores[v_ptn] = updated_score;
            size_t erased = ord_scores.erase({old_ptn_score, v_ptn});
            TORCH_CHECK(
                erased == 1,
                "add v from ", old_ptn, "->", v_ptn,
                ": should erase exactly 1 element but see ", erased
            );
            ord_scores.insert({updated_score, v_ptn});
            ++ptn_nodes[v_ptn];
            omp_unset_lock(&locks[label*16]);

            #ifndef NDEBUG
            std::ostringstream oss;
            oss << "[v=" << v << "]";
            check_consistency(ptn_balance_scores, ord_scores, oss.str());
            #endif
        } // labeled case ends
        // cleaning up
        // O(deg(v))
        for (auto p : ptn_to_check) ptn_neighbors[p] = 0;
        ptn_to_check.clear();
    }
    }

    return vector_to_tensor(node2ptn);
}


TORCH_LIBRARY(Fennel, m)
{
    m.def(
        "partition(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init) -> Tensor",
        partition);
    m.def(
        "partition_opt(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init, float? scan_thres) -> Tensor",
        partition_opt);
    m.def(
        "partition_parallel(Tensor rowptr, Tensor col, int k, Tensor? order, "
        "float gamma, float alpha, float slack, Tensor? init, float? scan_thres) -> Tensor",
        partition_par);
    m.def(
        "partition_strata(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, Tensor? init) -> Tensor",
        partition_stratified);
    m.def(
        "partition_strata_opt(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, Tensor? init, float? scan_thres) -> Tensor",
        partition_stratified_opt);
    m.def(
        "partition_strata_par(Tensor rowptr, Tensor col, int k, Tensor? labels, Tensor? order,"
        "float gamma, Tensor alphas, float slack, Tensor? init, float? scan_thres) -> Tensor",
        partition_stratified_par);
}

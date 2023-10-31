// fennel with weighted edges
#include <ctime>
#include <limits>
#include <vector>
#include <set>
#include <functional>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iterator>
#include <stdlib.h>
#include "fennel.hpp"

namespace fennel {

Tensor partition_weighted(
    Tensor ptr, Tensor idx, OptTensor weights, int64_t k, OptTensor node_order,
    double gamma, double alpha, double slack,
    OptTensor init_partition, torch::optional<double> scan_thres
) {
    if (!weights.has_value()) {
        return partition_opt(
            ptr, idx, k, node_order, gamma, alpha, slack, init_partition, scan_thres
        );
    }
    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    auto wgt_data = weights.value().data_ptr<float>();
    // hard cap for #nodes in each partition
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
    // auto ptn_sizes_tensor = init_ptn_tensor.to(torch::kDouble).histc(k+1, 0, k+1).to(torch::kInt64);
    auto ptn_sizes_tensor = init_ptn_tensor.bincount({}, k+1);
    TORCH_CHECK(ptn_sizes_tensor.sum().item().to<int64_t>() == n);
    auto ptn_nodes = tensor_to_vector<int64_t>(ptn_sizes_tensor);
    TORCH_CHECK(ptn_nodes.size() == k+1);
    auto node2ptn = Slice<int>::from_tensor(init_ptn_tensor).to_vec();
    TORCH_CHECK(node2ptn.size() == n);

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

    // scratchpad vectors
    std::vector<float> ptn_scores(k + 1, 0);
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
        auto w_adj = adj(ptr_data, wgt_data, v);
        for (int i = 0; i < v_adj.size(); ++i) {
            auto ngh = v_adj[i];
            auto wgt = w_adj[i];
            int ptn = node2ptn[ngh];
            if (ptn != k && ptn_scores[ptn] == 0) ptn_to_check.push_back(ptn);
            ptn_scores[ptn] += wgt;
        }
        // get the partition having the maximum score
        // O(deg(v))
        float current_max = min_score;
        if (ptn_to_check.size() >= k * scan_thres.value_or(1.0)) {
            for (size_t i = 0; i < k; ++i)
            {
                if (ptn_nodes[i] >= cap) continue;
                float ptn_score = ptn_scores[i] + ptn_balance_scores[i];
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
                float ptn_score = ptn_scores[p] + ptn_balance_scores[p];
                if (ptn_score > current_max) {
                    current_max = ptn_score;
                    v_ptn = p;
                }
            }
            auto score_pair = *ord_scores.begin();
            // an untouched partition has a larger score than current_max due to balancing
            if (v_ptn != score_pair.second && score_pair.first > current_max) {
                v_ptn = score_pair.second;
            }
        }

        // assign v to v_ptn
        // O(log(k))
        node2ptn[v] = v_ptn;
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
            std::fill(ptn_scores.begin(), ptn_scores.end(), 0);
        } else {
            for (auto p : ptn_to_check) ptn_scores[p] = 0;
        }
        ptn_to_check.clear();
    }

    return vector_to_tensor(node2ptn);
}

Tensor partition_stratified_balanced_weighted(
    Tensor ptr, Tensor idx, OptTensor weights, int64_t k, OptTensor node_order,
    double gamma, Tensor alphas, double balance_slack,
    Tensor stratify_labels, Tensor balance_labels,
    OptTensor init_partition, torch::optional<double> scan_thres
) {
    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    float *wgt_data = nullptr;
    const bool has_weights = weights.has_value();
    if (has_weights) {
        wgt_data = weights.value().data_ptr<float>();
    }

    // number of stratify labels
    TORCH_CHECK(stratify_labels.min().item().to<int>() >= 0, "Node labels can't be negative");
    TORCH_CHECK(stratify_labels.scalar_type() == torch::kInt);
    int SL = stratify_labels.max().item().to<int>() + 1;
    // number of balance labels
    TORCH_CHECK(balance_labels.min().item().to<int>() >= 0, "Node labels can't be negative");
    TORCH_CHECK(balance_labels.scalar_type() == torch::kInt);
    int BL = balance_labels.max().item().to<int>() + 1;

    // hard cap for #nodes in each partition
    int64_t cap = (int64_t)((float)n / k * balance_slack);
    // per-label cap for #nodes in each partition
    const std::vector<int> blabel2cap = tensor_to_vector<int>(
        balance_labels.bincount({}, BL).to(torch::kFloat)
                      .div_(k).multiply_(balance_slack)
                      .ceil_().to(torch::kInt)
    );

    // map: node -> label
    const auto slabels = Slice<int>::from_tensor(stratify_labels);
    const auto blabels = Slice<int>::from_tensor(balance_labels);

    // node streaming order
    // Tensor tensor_node_order = node_order.value_or(torch::randperm(n, torch::kInt64));
    Tensor tensor_node_order = node_order.value_or(torch::arange(n, torch::kInt64));
    auto node_stream = Slice<int64_t>::from_tensor(tensor_node_order);
    TORCH_CHECK(tensor_node_order.size(0) == n,
                "The provided node ordering should cover all nodes exactly once");

    // partition from scratch or from an initial partition
    if (init_partition.has_value())
        TORCH_CHECK(init_partition.value().size(0) == n,
                    "init_partition tensor should cover all nodes exactly once"
        );
    Tensor init_ptn_tensor = init_partition.value_or(torch::ones(n, torch::kInt32)*k);
    auto ptn_sizes_tensor = init_ptn_tensor.bincount({}, k+1);
    int64_t num_assigned = ptn_sizes_tensor.sum().item().to<int64_t>();
    TORCH_CHECK(num_assigned == n, num_assigned, " != ", n);
    // map: partition -> size
    auto ptn_sizes = tensor_to_vector<int64_t>(ptn_sizes_tensor);
    TORCH_CHECK(ptn_sizes.size() == k+1);
    // map: node -> partition
    auto node2ptn = tensor_to_vector<int>(init_ptn_tensor);
    TORCH_CHECK(node2ptn.size() == n);

    // map: stratify label -> (map: ptn -> size), we need them to compute label balance scores
    std::vector<std::vector<int64_t>> slabel2sizes(SL, std::vector<int64_t>(k+1, 0));
    for (int64_t i = 0; i < n; ++i) {
        int label = slabels[i];
        int ptn = node2ptn[i];
        slabel2sizes[label][ptn]++;
    }
    // map: balance label -> (map: ptn -> sizes), we need them to enforce the hard caps
    std::vector<std::vector<int64_t>> blabel2sizes(BL, std::vector<int64_t>(k+1, 0));
    for (int64_t i = 0; i < n; ++i) {
        int label = blabels[i];
        int ptn = node2ptn[i];
        blabel2sizes[label][ptn]++;
    }

    // map: stratify label -> alpha
    std::vector<float> slabel2alpha = tensor_to_vector<float>(alphas);
    // map: label -> (map: ptn -> score)
    // compute initial balance scores
    std::vector<std::vector<float>> slabel2scores(SL, std::vector<float>(k, 0));
    for (int label = 0; label < SL; ++label) {
        float alpha = slabel2alpha[label];
        std::transform(
            slabel2sizes[label].begin(), slabel2sizes[label].begin()+k, slabel2scores[label].begin(),
            [=](int64_t ptn_size) {
                return balance_score(ptn_size, gamma, alpha);
            }
        );
    }
    // map: label -> (balance_score -> partition)
    auto cmp_score = [&](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) {
        // we need to break ties when scores are equal
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second > rhs.second);
    };
    using rev_scoreset = std::set<std::pair<float, int>, decltype(cmp_score)>;
    std::vector<rev_scoreset> slabel2heap(SL, rev_scoreset(cmp_score));
    for (int label = 0; label < SL; ++label) {
        auto &scores = slabel2scores[label];
        auto &heap = slabel2heap[label];
        for (int i = 0; i < k; ++i) {
            heap.insert({scores[i], i});
        }
    }

    int64_t moved_nodes = 0;
    // balance_score(n, gamma, alphas.max().item().to<float>());
    const float min_score = std::numeric_limits<float>::lowest();
    std::vector<int> ptn_idx(k);
    std::iota(ptn_idx.begin(), ptn_idx.end(), 0);
    // scratchpad vectors
    std::vector<float> ptn_weights(k + 1, 0);
    std::vector<int> ptn_to_check; ptn_to_check.reserve(k);
    for (auto v : node_stream) {
        // random assign a partition if no partition is found
        int v_ptn = rand() % k;
        auto v_adj = adj(ptr_data, idx_data, v);
        auto w_adj = adj(ptr_data, wgt_data, v);
        int slabel = slabels[v];
        int blabel = blabels[v];
        // probe neighbors: O(deg(v))
        for (int i = 0; i < v_adj.size(); ++i) {
            auto ngh = v_adj[i];
            auto wgt = has_weights ? w_adj[i] : 1;
            int ptn = node2ptn[ngh];
            if (ptn != k && ptn_weights[ptn] == 0 && wgt != 0) ptn_to_check.push_back(ptn);
            ptn_weights[ptn] += wgt;
        }
        const bool scan_ptns = (ptn_to_check.size() >= k * scan_thres.value_or(1.0));

        // balance score per partition for current label
        std::vector<float> &ptn_balance_scores = slabel2scores[slabel];
        // size per partition for current stratify label
        std::vector<int64_t> &ptn_slabel_sizes = slabel2sizes[slabel];
        // ordered score per partition for current label
        rev_scoreset &ord_scores = slabel2heap[slabel];
        // alpha for the current label
        float alpha = slabel2alpha[slabel];
        // size per partition for current balance label
        std::vector<int64_t> &ptn_blabel_sizes = blabel2sizes[blabel];
        int64_t label_cap = blabel2cap[blabel];
        // remove v from its original partition
        // O(log(k))
        int old_ptn = node2ptn[v];

        if (old_ptn != k) {
            float old_ptn_score = ptn_balance_scores[old_ptn];
            float updated_score = balance_score(ptn_slabel_sizes[old_ptn]-1, gamma, alpha);
            // replace {old_score, ptn} with {new_score, ptn}
            size_t erased = ord_scores.erase({old_ptn_score, old_ptn});
            TORCH_CHECK(erased == 1,
                "remove v: should erase exactly 1 element but see ", erased
            );
            ord_scores.insert({updated_score, old_ptn});
            ptn_balance_scores[old_ptn] = updated_score;
            --ptn_slabel_sizes[old_ptn];
            --ptn_blabel_sizes[old_ptn];
            --ptn_sizes[old_ptn];
        }

        // get the partition having the maximum score
        // O(deg(v)+log(k))
        float current_max = min_score;
        auto &candidates = ptn_to_check;
        if (scan_ptns) candidates = ptn_idx;
        for (auto p : candidates) {
            if (ptn_sizes[p] >= cap) continue;
            if (ptn_blabel_sizes[p] >= label_cap) continue;
            float ptn_score = ptn_weights[p] + ptn_balance_scores[p];
            if (ptn_score > current_max) {
                current_max = ptn_score;
                v_ptn = p;
            }
        }
        // an untouched partition has larger score than current_max
        auto score_pair = *ord_scores.begin();
        // if (v_ptn != score_pair.second && score_pair.first > current_max)
        if (score_pair.first > current_max) {
            v_ptn = score_pair.second;
        }

        // add v to partition v_ptn
        node2ptn[v] = v_ptn;
        TORCH_CHECK(v_ptn != k);
        moved_nodes += (v_ptn != old_ptn);

        float old_ptn_score = ptn_balance_scores[v_ptn];
        float updated_score = balance_score(ptn_slabel_sizes[v_ptn]+1, gamma, alpha);
        ptn_balance_scores[v_ptn] = updated_score;
        size_t erased = ord_scores.erase({old_ptn_score, v_ptn});
        TORCH_CHECK(erased == 1,
            "add v from ", old_ptn, "->", v_ptn,
            ": should erase exactly 1 element but see ", erased
        );
        ord_scores.insert({updated_score, v_ptn});
        ++ptn_slabel_sizes[v_ptn];
        ++ptn_blabel_sizes[v_ptn];
        ++ptn_sizes[v_ptn];

        // cleaning up
        // O(deg(v))
        if (scan_ptns) {
            std::fill(ptn_weights.begin(), ptn_weights.end(), 0);
        } else {
            for (auto p : ptn_to_check) ptn_weights[p] = 0;
        }
        ptn_to_check.clear();
    }

    std::cout << "Total of " << moved_nodes << " nodes get re-assigned\n";
    return vector_to_tensor(node2ptn);
}

}

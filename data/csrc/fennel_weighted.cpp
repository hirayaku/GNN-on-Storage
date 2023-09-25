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

Tensor partition_stratified_weighted(
    Tensor ptr, Tensor idx, OptTensor weights, int64_t k, OptTensor labels, OptTensor node_order,
    double gamma, Tensor alphas, double slack, double label_slack,
    OptTensor init_partition, torch::optional<double> scan_thres
) {
    if (labels == torch::nullopt) {
        float alpha = alphas.index({0}).item().to<float>();
        return partition_weighted(
            ptr, idx, weights, k, node_order, gamma, alpha, slack, init_partition, scan_thres
        );
    }

    int64_t n = ptr.size(0) - 1;
    int64_t m = idx.size(0);
    if (weights == torch::nullopt) {
        weights = torch::make_optional(torch::ones(m, torch::kFloat));
        // return partition_stratified_opt(
        //     ptr, idx, k, labels, node_order, gamma, alphas, slack, label_slack,
        //     init_partition, scan_thres
        // );
    }

    auto ptr_data = ptr.data_ptr<int64_t>();
    auto idx_data = idx.data_ptr<int64_t>();
    auto wgt_data = weights.value().data_ptr<float>();

    // hard cap for #nodes in each partition
    int64_t cap = (int64_t)((float)n / k * slack);
    // per-label cap for #nodes in each partition
    auto label_tensor = labels.value();
    int NL = label_tensor.max().item().to<int>() + 1;
    const std::vector<long> label2cap = tensor_to_vector<int64_t>(
        label_tensor.bincount({}, NL).to(torch::kFloat)
                    .div_(k).multiply_(label_slack)
                    .ceil_().to(torch::kInt64)
    );
    // std::cout << "Cap per label: " << label2cap << "\n";
    // long train_cap = std::accumulate(label2cap.begin(), label2cap.end() - 1, 0);
    // std::cout << "Cap for training nodes per partition ("
    //     << k << ", "<< label_slack << "): " << train_cap << "\n";

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
    Tensor init_ptn_tensor = init_partition.value_or(torch::ones(n, torch::kInt32)*k);
    // auto ptn_sizes_tensor = init_ptn_tensor.to(torch::kDouble).histc(k+1, 0, k+1).to(torch::kInt64);
    auto ptn_sizes_tensor = init_ptn_tensor.bincount({}, k+1);
    int64_t num_assigned = ptn_sizes_tensor.sum().item().to<int64_t>();
    TORCH_CHECK(num_assigned == n, num_assigned, " != ", n);
    // map: partition -> size
    auto ptn_sizes = tensor_to_vector<int64_t>(ptn_sizes_tensor);
    TORCH_CHECK(ptn_sizes.size() == k+1);
    // map: node -> partition
    auto node2ptn = tensor_to_vector<int>(init_ptn_tensor);
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
    }

    // balance_score(n, gamma, alphas.max().item().to<float>());
    const float min_score = std::numeric_limits<float>::lowest();
    std::vector<int> ptn_idx(k);
    std::iota(ptn_idx.begin(), ptn_idx.end(), 0);
    // scratchpad vectors
    std::vector<float> ptn_weights(k + 1, 0);
    std::vector<int> ptn_to_check; ptn_to_check.reserve(k);

    int64_t processed = 0;
    for (auto v : node_stream)
    {
        processed++;
        if (processed % 10000000 == 0) {
            std::clog << processed / 1000000 << "M nodes assigned\n";
        }
        // random assign a partition if no partition is found
        int v_ptn = rand() % k;
        auto v_adj = adj(ptr_data, idx_data, v);
        auto w_adj = adj(ptr_data, wgt_data, v);
        int label = node2label[v];
        // probe neighbors: O(deg(v))
        for (int i = 0; i < v_adj.size(); ++i) {
            auto ngh = v_adj[i];
            auto wgt = w_adj[i];
            int ptn = node2ptn[ngh];
            if (ptn != k && ptn_weights[ptn] == 0 && wgt != 0) ptn_to_check.push_back(ptn);
            ptn_weights[ptn] += wgt;
        }
        const bool scan_ptns = (ptn_to_check.size() >= k * scan_thres.value_or(1.0));

        TORCH_CHECK(label >= 0, "node ", v, " has label < 0");
        if (label < 0) {
            // for nodes no need to balance, select the closest partition
            float current_max = 0;
            for (auto p : ptn_to_check) {
                if (ptn_weights[p] > current_max) {
                    current_max = ptn_weights[p];
                    v_ptn = p;
                }
            }
            node2ptn[v] = v_ptn;
        } else {
            // balance score per partition for current label
            std::vector<float> &ptn_balance_scores = label2scores[label];
            // size per partition for current label
            std::vector<int64_t> &ptn_label_sizes = label2sizes[label];
            // ordered score per partition for current label
            rev_scoreset &ord_scores = label2ord_scores[label];
            // alpha for the current label
            float alpha = label2alpha[label];
            int64_t label_cap = label2cap[label];
            // remove v from its original partition
            // O(log(k))
            int old_ptn = node2ptn[v];

            if (old_ptn != k) {
                float old_ptn_score = ptn_balance_scores[old_ptn];
                float updated_score = balance_score(ptn_label_sizes[old_ptn]-1, gamma, alpha);
                // replace {old_score, ptn} with {new_score, ptn}
                size_t erased = ord_scores.erase({old_ptn_score, old_ptn});
                TORCH_CHECK(
                    erased == 1,
                    "remove v: should erase exactly 1 element but see ", erased
                );
                ord_scores.insert({updated_score, old_ptn});
                ptn_balance_scores[old_ptn] = updated_score;
                --ptn_label_sizes[old_ptn];
                --ptn_sizes[old_ptn];
            }

            // get the partition having the maximum score
            // O(deg(v)+log(k))
            float current_max = min_score;
            auto &candidates = ptn_to_check;
            if (scan_ptns) candidates = ptn_idx;
            for (auto p : candidates) {
                if (ptn_sizes[p] >= cap) continue;
                if (ptn_label_sizes[p] >= label_cap) continue;
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

            float old_ptn_score = ptn_balance_scores[v_ptn];
            float updated_score = balance_score(ptn_label_sizes[v_ptn]+1, gamma, alpha);
            ptn_balance_scores[v_ptn] = updated_score;
            size_t erased = ord_scores.erase({old_ptn_score, v_ptn});
            TORCH_CHECK(
                erased == 1,
                "add v from ", old_ptn, "->", v_ptn,
                ": should erase exactly 1 element but see ", erased
            );
            ord_scores.insert({updated_score, v_ptn});
            ++ptn_label_sizes[v_ptn];
            ++ptn_sizes[v_ptn];

        } // labeled case ends

        // cleaning up
        // O(deg(v))
        if (scan_ptns) {
            std::fill(ptn_weights.begin(), ptn_weights.end(), 0);
        } else {
            for (auto p : ptn_to_check) ptn_weights[p] = 0;
        }
        ptn_to_check.clear();
    }

    // for (int label = 0; label < label2sizes.size(); label++) {
    //     std::cout << label << ") " <<  label2sizes[label] << '\n';
    // }

    return vector_to_tensor(node2ptn);
}

}

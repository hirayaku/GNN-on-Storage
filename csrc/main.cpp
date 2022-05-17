#include <iostream>
#include <tuple>
#include <thread>
#include <torch/torch.h>
#include "graph_io.hpp"
#include "graph_partition.hpp"
#include "utils.hpp"

using namespace gnnos;

#define RUN(fn) \
    std::cout << "[RUN]\t" #fn << "\n"; \
    fn(); \
    std::cout << "[PASS]\t" #fn << "\n";

static const TensorInfo products_options =
TensorOptions("/mnt/md0/inputs/ogbn_products/edge_index")
    .shape({2, 123718280})
    .itemsize(8);

void testCOOStore() {
    auto tensor = TensorStore::OpenForRead(products_options);

    gnnos::COOStore coo(
        tensor.slice(0, 1).flatten(),
        tensor.slice(1, 2).flatten(),
        2449029
    );

    gnnos::COOStore coo2 = coo.slice(0, 10);
    std::vector<int64_t> src, dst;
    std::tie(src, dst) = coo.accessor<int64_t>().slice(0, 10);
    for (int i = 0; i < 10; ++i) {
        std::cout << std::make_pair(src[i], dst[i]) << "\n";
    }

    // exception here
    coo.accessor<int64_t>().slice_put(src.data(), dst.data(), 0, 10);
}

void testCOOStoreOpen() {
    auto graph = TensorStore::OpenForRead(products_options);
    // exception here
    TensorStore::Open(graph.metadata().path("edge_index"));
}

void testCOOStoreCreateTemp() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto new_tensor = gnnos::TensorStore::CreateTemp(
        tensor.metadata().path("/mnt/md0/graphs/ogbn_products").offset(16)
    );

    COOStore coo(tensor.flatten(), 2449029);
    COOStore new_coo(new_tensor.flatten(), coo.num_nodes());
    CHECK_EQ(coo.num_edges(), new_coo.num_edges());

    auto edges = coo.accessor<int64_t>().slice(0, coo.num_edges());
    new_coo.accessor<int64_t>().slice_put(edges.first.data(), edges.second.data(),
        0, new_coo.num_edges());
    CHECK_EQ(coo.accessor<int64_t>().slice(0, 100), new_coo.accessor<int64_t>().slice(0, 100));

    auto new_tensor2 = gnnos::TensorStore::CreateTemp(
        tensor.metadata().path("/mnt/md0/graphs/ogbn_products").offset(8)
    );
    auto data2 = new_tensor2.accessor<int64_t>().slice(0, 8);
    // should be zeros, independent of new_tensor
    for (size_t i = 0; i < 8; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << "\n";
}

void testCOOStoreClone() {
    auto tensor = TensorStore::OpenForRead(products_options);
    LOG(INFO) << tensor.metadata();
    auto coo = COOStore(tensor.flatten(), 2449029);
    {
        LOG(INFO) << "start clone";
        auto coo_clone = coo.clone("edge_index", false);
        LOG(INFO) << coo_clone.metadata();
        LOG(INFO) << "finish clone";
    }

    auto tensor_clone = TensorStore::OpenForRead(
        tensor.metadata().path("edge_index")
    );
    auto coo_clone = COOStore(tensor.flatten(), 0);

    CHECK_EQ(coo.num_edges(), coo_clone.num_edges());
}

void testCOOStoreTraverse(size_t edge_block=1024) {
    // when edge_block == 1, IO bandwidth reduced to 1~2 MTEPS
    // when edge_block == 4096, IO bandwidth ~100 MTEPS
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    auto edges = coo.accessor<int64_t>();
    for (auto i = 0; i + edge_block < edges.size(); i += edge_block) {
        edges.slice(i, i+edge_block);
    }
}

void testCOOStorePartition1D() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom1D(coo, partition);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; ++i) {
        auto block = dcoo.coo_block(i);
        auto accessor = block.accessor<long>();
        for (int eid = 0; eid < accessor.size(); ++eid) {
            auto e = accessor[eid];
            CHECK_EQ(assigns_vec[e.first], i);
        }
    }
}

void testCOOStorePartition2D() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom2D(coo, partition);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; ++i) {
        for (int j = 0; j < psize; ++j) {
            int from = i, to = j;
            auto block = dcoo.coo_block(from * psize + to);
            auto accessor = block.accessor<long>();
            for (int eid = 0; eid < accessor.size(); ++eid) {
                auto e = accessor[eid];
                CHECK_EQ(assigns_vec[e.first], from);
                CHECK_EQ(assigns_vec[e.second], to);
            }
        }
    }
}

void testCOOToCSRStore() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    auto csr = CSRStore::NewFrom(coo);
}

void testCSRStoreOpen() {
    std::string folder = "/mnt/md0/inputs/oag-paper/";
    std::string ptr_file = folder + "graph.vertex.bin";
    std::string idx_file = folder + "graph.edge.bin";

    long num_nodes = 15257994;
    long num_edges = 220126508;
    auto ptr_store = TensorStore::OpenForRead(
        TensorOptions(ptr_file).shape({num_nodes+1}));
    auto idx_store = TensorStore::OpenForRead(
        TensorOptions(idx_file).shape({num_edges}));
    CSRStore csr{ptr_store, idx_store};
}

void testCSRToBCOO() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom2D(coo, partition);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; ++i) {
        for (int j = 0; j < psize; ++j) {
            int from = i, to = j;
            auto block = dcoo.coo_block(from * psize + to);
            auto accessor = block.accessor<long>();
            for (int eid = 0; eid < accessor.size(); ++eid) {
                auto e = accessor[eid];
                CHECK_EQ(assigns_vec[e.first], from);
                CHECK_EQ(assigns_vec[e.second], to);
            }
        }
    }
}

void testNodePartitions() {
    auto rand = torch::randint(10, {10}, torch::dtype(torch::kInt));
    auto p = NodePartitions::New(10, rand);
    LOG(INFO) << "Check NodePartitions";
    auto assigns = p.assignments().accessor<int, 1>();
    for (int i = 0; i < p.psize; ++i) {
        auto nodes = p[i];
        for (auto n : tensor_iter<int64_t>(nodes)) {
            CHECK_EQ(assigns[n], i);
        }
    }
}

void testBCOOSubgraph() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto bcoo = BCOOStore::PartitionFrom2D(coo, partition);
    LOG(INFO) << "get subgraphs...";
    bcoo.cluster_subgraph({1});
    bcoo.cluster_subgraph({100});
    bcoo.cluster_subgraph({0, 1});
    bcoo.cluster_subgraph({0, 1, 9, 127});
}

void testGather() {
    auto tensor = TensorStore::OpenForRead(products_options).flatten();
    GatherSlices(tensor, {{2, 4}, {6, 7}}, torch::kLong);
}

void getTorchInfo() {
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes": "no") << '\n';
    std::cout << torch::get_parallel_info();
}

void testTorchMM() {
	torch::Tensor tensor = torch::randn({2708, 1433});
	torch::Tensor weight = torch::randn({1433, 16});
	auto start = std::chrono::high_resolution_clock::now();
	tensor.mm(weight);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout<< tensor.sizes() << " x " << weight.sizes() << ": "
        << std::chrono::duration<double>(end - start).count() << "s\n";
}

int main() {
    torch::ShowLogInfoToStderr();

    // RUN(getTorchInfo);
    // RUN(testTorchMM);
    // RUN(testCOOStore);
    // RUN(testCOOStoreOpen);
    // RUN(testCOOStoreCreateTemp);
    // RUN(testCOOStoreClone);
    // RUN(testCOOStoreTraverse);
    // RUN(testNodePartitions);
    // RUN(testCOOStorePartition1D);
    // RUN(testCOOStorePartition2D);
    // RUN(testCOOToCSRStore);
    // RUN(testCSRToBCOO);
    // RUN(testBCOOSubgraph);
    RUN(testGather);

    return 0;
}

#include <iostream>
#include <tuple>
#include <thread>
#include <torch/torch.h>
#include "graph_io.hpp"
#include "graph_partition.hpp"
#include "utils.hpp"

using namespace gnnos;

static const TensorInfo products_options =
TensorOptions("/mnt/md0/graphs/ogbn_products/serialize/edge_index")
    .shape({2, 123718280})
    .itemsize(8);

void testCOOStore() {
    LOG(WARNING) << "testCOOStore";

    auto tensor = TensorStore::OpenForRead(products_options);

    gnnos::COOStore coo(
        tensor.slice(0, 1).flatten(),
        tensor.slice(1, 2).flatten(),
        2449029
    );
    gnnos::COOStore coo2(
        tensor.flatten(),
        2449029
    );
    std::vector<int64_t> src, dst;
    std::tie(src, dst) = coo2.accessor<int64_t>().slice(0, 10);
    for (int i = 0; i < 10; ++i) {
        std::cout << std::make_pair(src[i], dst[i]) << "\n";
    }

    // exception here
    // coo.accessor<int64_t>().slice_put(edges.first.data(), edges.second.data(), 0, 10);

}

void testCOOStoreOpen() {
    LOG(WARNING) << "testCOOStoreOpen";
    // exception here
    // TensorStore::OpenForRead(
    //     "/mnt/md0/graphs/ogbn_products/serialize/edge_index",
    //     TensorStore::option().shape({2, 123718280}).itemsize(8)
    // );
    // exception here
    // TensorStore::Open(
    //     "edge_index",
    //     TensorStore::option().shape({2, 123718280}).itemsize(8)
    // );
}

void testCOOStoreCreateTemp() {
    LOG(WARNING) << "testCOOStoreCreate";
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
    LOG(WARNING) << "testCOOStoreClone";
    auto tensor = TensorStore::OpenForRead(products_options);
    LOG(WARNING) << tensor.metadata();
    auto coo = COOStore(tensor.flatten(), 2449029);
    {
        LOG(WARNING) << "start clone";
        auto coo_clone = coo.clone("edge_index", false);
        LOG(WARNING) << coo_clone.metadata();
        LOG(WARNING) << "finish clone";
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
    LOG(WARNING) << "testCOOStoreClone";
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    auto edges = coo.accessor<int64_t>();
    for (auto i = 0; i + edge_block < edges.size(); i += edge_block) {
        edges.slice(i, i+edge_block);
    }
}

void testCOOStorePartition1D() {
    LOG(WARNING) << "testCOOStorePartition";
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto assigns = random_assignment(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom1D(coo, assigns, psize);

    // check BCOOStore consistency
    LOG(WARNING) << "Check BCOOStore";
    auto assigns_vec = assigns.accessor<int, 1>();
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
    LOG(WARNING) << "testCOOStorePartition2D";
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto assigns = random_assignment(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom2D(coo, assigns, psize);

    // check BCOOStore consistency
    LOG(WARNING) << "Check BCOOStore";
    auto assigns_vec = assigns.accessor<int, 1>();
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

void getTorchInfo() {
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "yes": "no") << '\n';
    std::cout << torch::get_parallel_info();
}

void testTorchTensor() {
    torch::Tensor tensor = torch::rand({3, 3});
    std::cout << tensor << std::endl;
    // assert foo is 2-dimensional and holds floats.
    auto foo_a = tensor.accessor<float,2>();
    float trace = 0;

    for(int i = 0; i < foo_a.size(0); i++) {
        // use the accessor foo_a to get tensor data.
        trace += foo_a[i][i];
        for (int j = 0; j < foo_a.size(1); ++j)
            foo_a[i][j] = j;
    }
    std::cout << "trace: " << trace << "\n";
    std::cout << tensor << std::endl;
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
    // getTorchInfo();
    // testTorchMM();
    // testCOOStore();
    // testCOOStoreOpen();
    // testCOOStoreCreateTemp();
    // testCOOStoreClone();
    // testCOOStoreTraverse();
    testCOOStorePartition1D();
    // testCOOStorePartition2D();

    return 0;
}

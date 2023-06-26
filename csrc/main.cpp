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
    .shape({2, 123718280}).dtype(torch::kLong);

static std::vector<long> degreesOf(const COOStore &coo) {
    std::vector<long> degrees(coo.num_nodes());
    auto accessor = coo.accessor<long>();
    std::vector<long> src, dst;
    std::tie(src, dst) = accessor.slice(0, coo.num_edges());
    for (const auto id : dst) {
        degrees[id]++;
    }
    return degrees;
}

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
    TensorStore::OpenForRead(graph.metadata().path("edge_index"));
}

void testCOOStoreCreateTemp() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto new_tensor = gnnos::TensorStore::CreateTemp(
        tensor.metadata().path(TMPDIR).offset(16)
    );

    COOStore coo(tensor.flatten(), 2449029);
    COOStore new_coo(new_tensor.flatten(), coo.num_nodes());
    TORCH_CHECK_EQ(coo.num_edges(), new_coo.num_edges());

    auto edges = coo.accessor<int64_t>().slice(0, coo.num_edges());
    new_coo.accessor<int64_t>().slice_put(edges.first.data(), edges.second.data(),
        0, new_coo.num_edges());
    TORCH_CHECK_EQ(coo.accessor<int64_t>().slice(0, 100), new_coo.accessor<int64_t>().slice(0, 100));

    auto new_tensor2 = gnnos::TensorStore::CreateTemp(
        tensor.metadata().path(TMPDIR).offset(8)
    );
    auto data2 = new_tensor2.accessor<int64_t>().slice(0, 8);
    // should be zeros, independent of new_tensor
    for (size_t i = 0; i < 8; ++i) {
        std::cout << data2[i] << " ";
    }
    std::cout << "\n";
}

/*
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

    TORCH_CHECK_EQ(coo.num_edges(), coo_clone.num_edges());
    auto edges = coo.slice(100, 200).tensor();
    auto clone_edges = coo.slice(100, 200).tensor();
    std::cout << "Comparison result: "
        << (std::get<0>(edges) == std::get<0>(clone_edges)).all() << "\n";
    std::cout << "Comparison result: "
        << (std::get<1>(edges) == std::get<1>(clone_edges)).all() << "\n";
}
*/

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
    auto bcoo = BCOOStore::PartitionFrom1D(coo, partition);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    auto degrees_coo = degreesOf(coo);
    auto degrees_bcoo = degreesOf(bcoo);
    for (long vid = 0; vid < coo.num_nodes(); ++vid) {
        TORCH_CHECK_EQ(degrees_coo[vid], degrees_bcoo[vid]);
    }

    LOG(INFO) << "Degrees match; check COO blocks randomly next";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; i += psize / 100 + 1) {
        auto block = bcoo.coo_block(i);
        auto accessor = block.accessor<long>();
        for (int eid = 0; eid < accessor.size(); ++eid) {
            auto e = accessor[eid];
            TORCH_CHECK_EQ(assigns_vec[e.first], i);
        }
    }
}

void testCOOStorePartition2D() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto bcoo = BCOOStore::PartitionFrom2D(coo, partition);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    auto degrees_coo = degreesOf(coo);
    auto degrees_bcoo = degreesOf(bcoo);
    for (long vid = 0; vid < coo.num_nodes(); ++vid) {
        TORCH_CHECK_EQ(degrees_coo[vid], degrees_bcoo[vid]);
    }

    LOG(INFO) << "Degrees match; check COO blocks randomly next";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; i += psize / 10 + 1) {
        for (int j = 0; j < psize; j += psize / 10 + 1) {
            int from = i, to = j;
            auto block = bcoo.coo_block(from * psize + to);
            auto accessor = block.accessor<long>();
            for (int eid = 0; eid < accessor.size(); ++eid) {
                auto e = accessor[eid];
                TORCH_CHECK_EQ(assigns_vec[e.first], from);
                TORCH_CHECK_EQ(assigns_vec[e.second], to);
            }
        }
    }
}

void testSaveCOOStore() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 128;
    auto partition = random_partition(coo, psize);
    auto dcoo = BCOOStore::PartitionFrom2D(coo, partition);
    long num_nodes;
    TensorInfo src_info, dst_info;
    std::tie(num_nodes, src_info, dst_info) = save_COOStore(dcoo, products_options.path() + ".p");
    auto coo2 = COOStore(TensorStore::OpenForRead(src_info), TensorStore::OpenForRead(dst_info),
        num_nodes);
    auto dcoo2 = BCOOStore(coo2, dcoo.edge_pos(), partition, P_2D);

    // check BCOOStore consistency
    LOG(INFO) << "Check BCOOStore";
    const auto assigns_vec = partition.assignments().accessor<int, 1>();
    for (int i = 0; i < psize; ++i) {
        for (int j = 0; j < psize; ++j) {
            int from = i, to = j;
            auto block = dcoo2.coo_block(from * psize + to);
            auto accessor = block.accessor<long>();
            for (int eid = 0; eid < accessor.size(); ++eid) {
                auto e = accessor[eid];
                TORCH_CHECK_EQ(assigns_vec[e.first], from);
                TORCH_CHECK_EQ(assigns_vec[e.second], to);
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
                TORCH_CHECK_EQ(assigns_vec[e.first], from);
                TORCH_CHECK_EQ(assigns_vec[e.second], to);
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
            TORCH_CHECK_EQ(assigns[n], i);
        }
    }
}

void testBCOOSubgraph() {
    auto tensor = TensorStore::OpenForRead(products_options);
    auto coo = COOStore(tensor.flatten(), 2449029);
    int psize = 8;
    auto partition = random_partition(coo, psize);
    auto bcoo_2d = BCOOStore::PartitionFrom2D(coo, partition);
    auto bcoo_1d = BCOOStore::PartitionFrom1D(coo, partition);

    auto edge_pos = bcoo_2d.edge_pos();
    for (int pid = 0; pid < psize; ++pid) {
        auto degrees_1d = degreesOf(bcoo_1d.coo_block(pid));
        auto bcoo_2d_blocks = bcoo_2d.slice(
            edge_pos[pid*psize].item<long>(),
            edge_pos[pid*psize+psize].item<long>()
        );
        auto degrees_2d = degreesOf(bcoo_2d_blocks);
        LOG(INFO) << "Checking Partition " << pid;
        for (long vid = 0; vid < coo.num_nodes(); ++vid) {
            TORCH_CHECK_EQ(degrees_1d[vid], degrees_2d[vid]) << "vid=" << vid;
        }
        TORCH_CHECK_EQ(std::accumulate(degrees_1d.begin(), degrees_1d.end(), 0),
            bcoo_1d.coo_block(pid).num_edges());
    }

    LOG(INFO) << "Getting subgraphs...";
    auto sg_2d = bcoo_2d.cluster_subgraph({0, 1});
    auto sg_1d = bcoo_1d.cluster_subgraph({0, 1});
    TORCH_CHECK_EQ(std::get<0>(sg_2d).numel(), std::get<0>(sg_1d).numel());
    TORCH_CHECK_EQ(std::get<1>(sg_2d).numel(), std::get<1>(sg_1d).numel());

    // bcoo_2d.cluster_subgraph({1});
    // bcoo_2d.cluster_subgraph({100});
    // bcoo_2d.cluster_subgraph({0, 1, 9, 127});
}

void testGather() {
    auto tensor = TensorStore::OpenForRead(products_options).flatten();
    std::cout << "Gathered:\n" << GatherSlices(tensor, {{2, 4}, {123718280+2, 123718280+4}});
}

void testStoreIntTensor() {
    long num_edges = 2595497852;
    const TensorInfo options =
    TensorOptions("/mnt/md0/inputs/mag240m/graph.edge.bin")
        .shape({num_edges}).dtype(torch::kInt);
    auto store = TensorStore::OpenForRead(options);
    auto last_10 = store.slice(num_edges-10, num_edges).tensor();
    std::cout << last_10 << "\n";

    size_t sz = store.numel() * store.itemsize();
    int *buf = (int*)(new char[sz]);
    ssize_t nbytes = store.pread(buf, sz, 0);
    std::cout << store.metadata() << "\n";
    std::cout << "request " << sz << " Bytes; read out " << nbytes << " Bytes\n";
    for (size_t i = num_edges - 10; i < num_edges; ++i) {
        std::cout << buf[i] << " ";
    }
    std::cout << '\n';
    delete []buf;
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
    c10::InferenceMode guard(true);

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
    // RUN(testSaveCOOStore);
    // RUN(testCOOToCSRStore);
    // RUN(testCSRToBCOO);
    RUN(testBCOOSubgraph);
    // RUN(testGather);
    // RUN(testStoreIntTensor);

    return 0;
}

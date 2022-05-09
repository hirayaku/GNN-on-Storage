#include "graph_partition.hpp"

namespace gnnos {

torch::Tensor random_assignment(const COOStore &coo, int psize) {
    ssize_t num_nodes = coo.num_nodes();
    return torch::randint(psize, {num_nodes}, torch::TensorOptions(torch::kInt));
}

torch::Tensor go_assignment(const COOStore &coo, int psize) {
    auto rand_assigns = random_assignment(coo, psize);
    BCOOStore rand_dcoo = BCOOStore::PartitionFrom1D(coo, rand_assigns, psize);
}

}

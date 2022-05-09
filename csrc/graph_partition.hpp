#ifndef GNNOS_GRAPH_PARTITION_HPP_
#define GNNOS_GRAPH_PARTTION_HPP_

#include <torch/torch.h>
#include "graph_io.hpp"

namespace gnnos {

torch::Tensor random_assignment(const COOStore &coo, int psize);
torch::Tensor go_assignment(const COOStore &coo, int psize);

}   // ns gnnos

#endif

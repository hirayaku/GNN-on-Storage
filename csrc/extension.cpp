#include <unordered_map>
#include "tensor_io.hpp"
#include "graph_io.hpp"
#include "graph_partition.hpp"
#include <torch/extension.h>
#include <torch/python.h>

using namespace gnnos;

#define GNNOS_FORALL_LIBTORCH_SCALAR_TYPES(_) \
  _(Byte, uint8)                                \
  _(Char, int8)                                 \
  _(Short, int16)                               \
  _(Int, int32)                                     \
  _(Long, int64)                                \
  _(Half, float16)                               \
  _(Float, float32)                                 \
  _(Double, float64)                               \
  _(Bool, bool)                                  \
  _(BFloat16, bfloat16)

static py::object pytorch = py::module_::import("torch");

// from libtorch dtype to pytorch dtype
static py::object pytorch_dtype(torch::ScalarType dtype) {
#define DEFINE_CASE(libtype, pytype) \
    case torch::k##libtype: \
        return pytorch.attr(#pytype); \

    switch (dtype) {
        GNNOS_FORALL_LIBTORCH_SCALAR_TYPES(DEFINE_CASE)
        default: throw std::invalid_argument("Unknown libtorch dtype");
    }
#undef DEFINE_CASE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verbose", []() { torch::ShowLogInfoToStderr(); });
    m.def("get_tmp_dir", []() { return TMPDIR; });
    m.def("set_tmp_dir", [&](std::string dir) { TMPDIR = dir; }, py::arg("dir"));
    m.def("get_io_threads", []() { return IO_THREADS; });
    m.def("set_io_threads", [&](int num_threads) { IO_THREADS = num_threads; },
        py::arg("threads"));

    // Tensor methods
    py::class_<TensorInfo>(m, "TensorInfo")
        .def(py::init<>())
        .def_property_readonly(
            "path", py::overload_cast<>(&TensorInfo::path, py::const_))
        .def_property_readonly(
            "shape", py::overload_cast<>(&TensorInfo::shape, py::const_))
        .def_property_readonly("dtype", [](const TensorInfo &self) {
            return pytorch_dtype(self.dtype());
        })
        .def_property_readonly(
            "offset", py::overload_cast<>(&TensorInfo::offset, py::const_))
        .def("with_path", py::overload_cast<std::string>(&TensorInfo::path))
        .def("with_shape", py::overload_cast<c10::IntArrayRef>(&TensorInfo::shape))
        .def("with_dtype", [](TensorInfo &tinfo, py::object dtype) {
            return tinfo.dtype(torch::python::detail::py_object_to_dtype(dtype));
        }, py::arg("dtype"))
        .def("with_offset", py::overload_cast<long>(&TensorInfo::offset))
        .def("__repr__", [](const TensorInfo &tinfo) {
            std::stringstream ss;
            ss << tinfo;
            return ss.str();
        });

    m.def("options",
        py::overload_cast<std::string>(&TensorOptions),
        "Create a default TensorInfo for a TensorStore",
        py::arg("path") = "");

    py::class_<TensorStore>(m, "TensorStore")
        .def(py::init<>())
        .def_property_readonly("numel", &TensorStore::numel)
        .def_property_readonly("shape", &TensorStore::shape)
        .def_property_readonly("metadata", &TensorStore::metadata)
        .def("element_size", &TensorStore::itemsize)
        .def("reshape", &TensorStore::reshape, py::arg("new_shape"))
        .def("flatten", &TensorStore::flatten)
        .def("__getitem__", [](const TensorStore &store, const py::slice &slice) {
                size_t start = 0, stop = 0, step = 0, slicelen = 0;
                if (!slice.compute(store.shape()[0], &start, &stop, &step, &slicelen))
                    throw py::value_error(
                        "Invalid slice argument: " + slice.str().cast<std::string>());
                if (step > 1)
                    throw py::value_error(
                        "Does not support non-consecutive slicing now");
                return store.slice(start, stop).tensor();
            }, "get TensorStore[slice]", py::arg("idx"))
        .def("__getitem__", &TensorStore::at, "get TensorStore[idx]", py::arg("idx"))
        .def("tensor", py::overload_cast<>(&TensorStore::tensor, py::const_),
            "read TensorStore into an in-memory Tensor")
        .def("tensor", [](const TensorStore &self, py::object dtype) {
                return self.tensor(torch::python::detail::py_object_to_dtype(dtype));
            },"read TensorStore into an in-memory Tensor and cast into dtype",
            py::arg("dtype"))
        .def("gather", &GatherSlices, "gather slices of stores into a torch Tensor",
            py::arg("slices"))
        .def("save", &TensorStore::save_to,
            "save TensorStore to file", py::arg("path"));

    m.def("tensor_store",
        [](const TensorInfo &tinfo, std::string flags, bool temp) {
            if (!temp) {
                if (flags == "r") {
                    return TensorStore::OpenForRead(tinfo);
                } else if (flags == "r+") {
                    return TensorStore::Open(tinfo);
                } else if (flags == "x") {
                    return TensorStore::Create(tinfo);
                } else {
                    throw std::invalid_argument("Invalid flags: " + flags);
                }
            } else {
                return TensorStore::CreateTemp(tinfo);
            }
        },
        "Create new TensorStore",
        py::arg("TensorInfo"), py::arg("flags") = "r", py::arg("temp") = false);

    m.def("from_tensor",
        py::overload_cast<const torch::Tensor&, std::string>(&TensorStore::NewFrom),
        "save the torch Tensor into TensorStore", py::arg("Tensor"), py::arg("path"));

    m.def("gather_slices", &GatherSlices,
        "gather store slices into a torch Tensor",
        py::arg("TensorStore"), py::arg("ranges"));

    m.def("gather_tensor_slices", &GatherTensorSlices,
        "gather tensor slices into a torch Tensor",
        py::arg("Tensor"), py::arg("ranges"));

    m.def("shuffle_store", &ShuffleStore, "shuffle store elements based on clusters",
        py::arg("TensorStore"), py::arg("TensorStore"), py::arg("shuffled_ids"));


    // COOStore, CSRStore

    py::class_<COOStore>(m, "COOStore")
        .def(py::init<>())
        .def(py::init<const TensorStore &, long>())
        .def(py::init<TensorStore, TensorStore, long>())
        .def_property_readonly("num_nodes", &COOStore::num_nodes)
        .def_property_readonly("num_edges", &COOStore::num_edges)
        .def_property_readonly("metadata", &COOStore::metadata)
        .def_property_readonly("src", [](const COOStore &self) { return self.src_store; })
        .def_property_readonly("dst", [](const COOStore &self) { return self.dst_store; })
        .def("slice", py::overload_cast<long, long>(&COOStore::slice, py::const_),
            "COOStore[start, end)", py::arg("start"), py::arg("end"))
        .def("tensor", &COOStore::tensor)
        .def("save", [](const COOStore &self, std::string path) {
            return save_COOStore(self, path);
        },"save COOStore to file", py::arg("path"));

    py::class_<CSRStore>(m, "CSRStore")
        .def(py::init<TensorStore, TensorStore>())
        .def_property_readonly("num_nodes", &CSRStore::num_nodes)
        .def_property_readonly("num_edges", &CSRStore::num_edges)
        .def_property_readonly("metadata", &CSRStore::metadata)
        .def("tensor", &CSRStore::tensor)
        .def("save", [](const CSRStore &self, std::string path) {
            return save_CSRStore(self, path);
        },"save CSRStore to file", py::arg("path"))
        .def("neighbors", &CSRStore::out_neighbors,
            "get neighbor nodes", py::arg("nid"))
        .def_static("from_coo", &CSRStore::NewFrom, "convert COOStore to CSRStore",
            py::arg("COOStore"));


    /*
    // Graph Partitioning

    py::class_<NodePartitions>(m, "NodePartitions")
        .def_readonly("psize", &NodePartitions::psize)
        .def("assignments", &NodePartitions::assignments)
        .def("pos", &NodePartitions::pos, py::arg("idx"))
        .def("size", &NodePartitions::size, py::arg("idx"))
        .def("__getitem__", &NodePartitions::operator[],
            "get the node partition tensor", py::arg("idx"))
        .def("nodes", &NodePartitions::nodes);

    m.def("node_partitions", &NodePartitions::New, "create a NodePartition object",
        py::arg("psize"), py::arg("assignments"));

    m.def("random_partition",
        py::overload_cast<const CSRStore&, int>(&random_partition),
        "generate a random node paritioning",
        py::arg("CSRStore"), py::arg("psize"));

    m.def("random_partition",
        py::overload_cast<const COOStore&, int>(&random_partition),
        "generate a random node paritioning",
        py::arg("COOStore"), py::arg("psize"));

    m.def("good_partition", &good_partition,
        "generate a good node paritioning", py::arg("CSRStore"), py::arg("psize"));

    py::enum_<PartitionType>(m, "PartitionType")
        .value("P_1D", PartitionType::P_1D)
        .value("P_2D", PartitionType::P_2D);

    py::class_<BCOOStore, COOStore>(m, "BCOOStore")
        .def(py::init<>())
        .def(py::init<COOStore, torch::Tensor, NodePartitions, PartitionType>())
        .def_property_readonly("psize", &BCOOStore::psize)
        .def_property_readonly("num_blocks", &BCOOStore::num_blocks)
        .def("edge_pos", &BCOOStore::edge_pos)
        .def("__getitem__", &BCOOStore::coo_block, "get the coo block", py::arg("idx"))
        .def("subgraph", &BCOOStore::cluster_subgraph,
            "get the subgraph induced from node clusters", py::arg("clusters"));

    m.def("partition_coo_1d", &BCOOStore::PartitionFrom1D,
        "1D partitioning COOStore", py::arg("COOStore"), py::arg("partition"));

    m.def("partition_coo_2d",
        py::overload_cast<const COOStore &, NodePartitions>(&BCOOStore::PartitionFrom2D),
        "2D partitioning COOStore", py::arg("COOStore"), py::arg("partition"));

    m.def("partition_csr_2d",
        [](const CSRStore &csr, NodePartitions p) {
            auto ptr_sz = csr.ptr_store.itemsize();
            auto idx_sz = csr.idx_store.itemsize();
            if (ptr_sz == 4 && idx_sz == 4) {
                return BCOOStore::PartitionFrom2D<int, int>(csr, p);
            } else if (ptr_sz == 8 && idx_sz == 4) {
                return BCOOStore::PartitionFrom2D<long, int>(csr, p);
            } else if (ptr_sz == 8 && idx_sz == 8) {
                return BCOOStore::PartitionFrom2D<long, long>(csr, p);
            } else {
                throw std::runtime_error("Invalid ptr_sz & idx_sz combination");
            }
        },
        "2D partitioning COOStore", py::arg("COOStore"), py::arg("partition"));
    */

}

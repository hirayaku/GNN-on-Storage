#include <unordered_map>
#include "tensor_io.hpp"
#include "graph_io.hpp"
#include "graph_partition.hpp"
#include <torch/extension.h>

using namespace gnnos;

#define GNNOS_FORALL_PYTORCH_SCALAR_TYPES(_) \
  _(uint8, Byte)                                \
  _(int8, Char)                                 \
  _(int16, Short)                               \
  _(short, Short)                               \
  _(int, Int)                                     \
  _(int32, Int)                                     \
  _(int64, Long)                                \
  _(long, Long)                                \
  _(half, Half)                               \
  _(float16, Half)                               \
  _(float, Float)                                 \
  _(float32, Float)                                 \
  _(double, Double)                               \
  _(float64, Double)                               \
  _(bool, Bool)                                  \
  _(bfloat16, BFloat16) 

static py::object pytorch = py::module_::import("torch");
static py::object pytorch_dtype = pytorch.attr("dtype");

// from pytorch dtype to libtorch dtype
static torch::ScalarType libtorch_dtype(py::object dtype) {
    auto dtype_str = py::repr(dtype);
    TORCH_CHECK(py::isinstance(dtype, pytorch_dtype),
        "Unknown dtype for gnnos: ", dtype_str);

#define DEFINE_ITEM(pytype, libtype) \
    { py::repr(pytorch.attr(#pytype)), torch::k##libtype }, \

    static std::unordered_map<std::string, torch::ScalarType> lookup {
        GNNOS_FORALL_PYTORCH_SCALAR_TYPES(DEFINE_ITEM)
    };
#undef DEFINE_ITEM

    auto iter = lookup.find(py::repr(dtype));
    TORCH_CHECK(iter != lookup.end(), "Unsupported dtype for gnnos: ", dtype_str);
    return iter->second;
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
        .def_property_readonly(
            "dtype", py::overload_cast<>(&TensorInfo::dtype, py::const_))
        .def_property_readonly(
            "offset", py::overload_cast<>(&TensorInfo::offset, py::const_))
        .def("with_path", py::overload_cast<std::string>(&TensorInfo::path))
        .def("with_shape", py::overload_cast<c10::IntArrayRef>(&TensorInfo::shape))
        .def("with_dtype", [](TensorInfo &tinfo, py::object dtype) {
            return tinfo.dtype(libtorch_dtype(dtype));
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
        .def_property_readonly("metadata", &TensorStore::metadata)
        .def("reshape", &TensorStore::reshape, py::arg("new_shape"))
        .def("flatten", &TensorStore::flatten)
        .def("slice", py::overload_cast<long, long>(&TensorStore::slice, py::const_),
            "TensorStore[start, end)", py::arg("start"), py::arg("end"))
        .def("__getitem__", &TensorStore::at, "get TensorStore[idx]", py::arg("idx"))
        .def("tensor", py::overload_cast<>(&TensorStore::tensor, py::const_),
            "read TensorStore into an in-memory Tensor")
        .def("tensor", [](const TensorStore &self, py::object dtype) {
                return self.tensor(libtorch_dtype(dtype));
            },"read TensorStore into an in-memory Tensor and cast into dtype",
            py::arg("dtype"))
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

    m.def("gather_slices", &GatherSlices,
        "gather store slices into a torch Tensor",
        py::arg("TensorStore"), py::arg("ranges"));

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
        .def("slice", py::overload_cast<long, long>(&COOStore::slice, py::const_),
            "COOStore[start, end)", py::arg("start"), py::arg("end"))
        .def("tensor", &COOStore::tensor);

    py::class_<CSRStore>(m, "CSRStore")
        .def(py::init<TensorStore, TensorStore>())
        .def_property_readonly("num_nodes", &CSRStore::num_nodes)
        .def_property_readonly("num_edges", &CSRStore::num_edges)
        .def_property_readonly("metadata", &CSRStore::metadata)
        .def("tensor", &CSRStore::tensor)
        .def("neighbors", &CSRStore::out_neighbors,
            "get neighbor nodes", py::arg("nid"))
        .def_static("from_coo", &CSRStore::NewFrom, "convert COOStore to CSRStore",
            py::arg("COOStore"));


    // Graph Partitioning

    py::class_<NodePartitions>(m, "NodePartitions")
        .def(py::init<>())
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

    m.def("go_partition", py::overload_cast<const CSRStore&, int>(&go_partition),
        "generate a good node paritioning", py::arg("CSRStore"), py::arg("psize"));

    py::class_<BCOOStore, COOStore>(m, "BCOOStore")
        .def(py::init<>())
        .def_property_readonly("psize", &BCOOStore::psize)
        .def_property_readonly("num_blocks", &BCOOStore::num_blocks)
        .def("__getitem__", &BCOOStore::coo_block, "get the coo block", py::arg("idx"))
        .def("subgraph", &BCOOStore::cluster_subgraph,
            "get the subgraph induced from node clusters", py::arg("clusters"))

        .def_static("from_coo_1d", &BCOOStore::PartitionFrom1D,
            "1D partitioning COOStore", py::arg("COOStore"), py::arg("partition"))

        .def_static("from_coo_2d",
            py::overload_cast<const COOStore &, NodePartitions>(&BCOOStore::PartitionFrom2D),
            "2D partitioning COOStore", py::arg("COOStore"), py::arg("partition"))

        .def_static("from_csr_2d",
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

}

#include "tensor_io.hpp"
#include "graph_io.hpp"
#include "graph_partition.hpp"
#include <torch/extension.h>

using namespace gnnos;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Tensor methods
    py::class_<TensorInfo>(m, "TensorInfo")
        .def(py::init<>())
        .def("path", py::overload_cast<>(&TensorInfo::path, py::const_))
        .def("shape", py::overload_cast<>(&TensorInfo::shape, py::const_))
        .def("itemsize", py::overload_cast<>(&TensorInfo::itemsize, py::const_))
        .def("offset", py::overload_cast<>(&TensorInfo::offset, py::const_))
        .def("path", py::overload_cast<std::string>(&TensorInfo::path))
        .def("shape", py::overload_cast<c10::IntArrayRef>(&TensorInfo::shape))
        .def("itemsize", py::overload_cast<int>(&TensorInfo::itemsize))
        .def("offset", py::overload_cast<long>(&TensorInfo::offset))
        .def("__repr__", [](const TensorInfo &tinfo) {
            std::stringstream ss;
            ss << tinfo;
            return ss.str();
        });

    m.def("tensor_options",
        py::overload_cast<std::string>(&TensorOptions),
        "Create a default TensorInfo for a TensorStore",
        py::arg("path") = "");

    py::class_<TensorStore>(m, "TensorStore")
        .def(py::init<>())
        .def("numel", &TensorStore::numel)
        .def("metadata", &TensorStore::metadata)
        .def("reshape", &TensorStore::reshape, py::arg("new_shape"))
        .def("flatten", &TensorStore::flatten)
        .def("slice", py::overload_cast<long, long>(&TensorStore::slice, py::const_),
            "TensorStore[start, end)", py::arg("start"), py::arg("end"))
        .def("tensor", &TensorStore::tensor,
            "read TensorStore into an in-memory torch Tensor")
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

    m.def("gather_slices", &GatherSlices, "gather store slices into a torch Tensor",
        py::arg("TensorStore"), py::arg("ranges"));

    m.def("shuffle_store", &ShuffleStore, "shuffle store elements based on clusters",
        py::arg("TensorStore"), py::arg("TensorStore"), py::arg("shuffled_ids"));

    // Graph methods
    py::class_<NodePartitions>(m, "NodePartitions")
        .def(py::init<>())
        .def_readonly("psize", &NodePartitions::psize)
        .def("__getitem__", &NodePartitions::operator[],
            "return the node partition tensor", py::arg("idx"));

    m.def("node_partitions", &NodePartitions::New, "create a NodePartition object",
        py::arg("psize"), py::arg("assignments"));
}

#include <cstdint>
#include <memory>
#include <torch/extension.h>
#include <torch/custom_class.h>

#include <c10/core/Allocator.h>
#include <ATen/MapAllocator.h>
#include <c10/core/CPUAllocator.h>
#include "ext_ops.hpp"

using Allocator = c10::Allocator;

// A wrapper for the static allocator objects, such as DefaultCPUAllocator
// these objects can't be deleted (because they are static) when out of scope
class StaticAllocatorWrapper: public Allocator {
    Allocator *allocator_ = nullptr;
  public:
    StaticAllocatorWrapper(Allocator *allocator): allocator_(allocator) {}
    // don't free pointers to static objects
    ~StaticAllocatorWrapper() {}
    c10::DataPtr allocate(size_t n) const override {
        return allocator_->allocate(n);
    }
    c10::DeleterFnPtr raw_deleter() const {
        return allocator_->raw_deleter();
    }
    static std::shared_ptr<Allocator>
    New(Allocator *allocator) {
        return std::make_shared<StaticAllocatorWrapper>(allocator);
    }
};

// MapAllocator does NOT sub-class Allocator.
// It doesn't implement `allocate`, so you cannot reuse a MapAllocator to allocate new memory.
// Instead, we could use MapAllocator::makeDataPtr to get allocated memory in DataPtr.
// DataPtr obj created this way is attached with a unqiue MapAllocator obj which provides
// the deleter for the allocated memory (munmap).
// using MapAllocator = at::MapAllocator;

// An alternative MapAllocator which cleans up (reclaims) the mmapped area
// with MADV_DONTNEED when the allocated tensor is freed
// class CleanMapAllocator : public at::MapAllocator;

class CustomMapAllocator : public Allocator {
    std::string filename;
    int flags;
    size_t size_;
  public:
    // TODO: file offsets
    CustomMapAllocator(std::string filename, int flags, size_t map_size=-1L)
    : filename(filename), flags(flags), size_(map_size) {}
    void close() {
        // std::clog << "CustomMapAllocator destroyed" << std::endl;
    }
    ~CustomMapAllocator() {
        this->close();
    }
    size_t size() const { return size_; }
    // creates new mmap/shm files on demand
    c10::DataPtr allocate(size_t alloc_size) const override {
        TORCH_CHECK(
            this->size() >= alloc_size, "trying to alloc more than reserved: ",
            alloc_size, " vs. ", this->size()
        );
        size_t actual_size;
        std::string fname = filename;
        auto data_ptr = at::MapAllocator::makeDataPtr(filename, flags, size_, &actual_size);
        if (actual_size != size_) {
            TORCH_WARN("Requested ", size_, " bytes but allocated ", actual_size);
        }
        return data_ptr;
    }
    static std::shared_ptr<Allocator>
    New(std::string filename, int flags, size_t size) {
        return std::make_shared<CustomMapAllocator>(filename, flags, size);
    }
};

/*
// for "file_system" sharing strategy
class CustomLibshmAllocator : public Allocator {
    std::string filename;
    int flags;
    size_t size_;
  public:
    CustomLibshmAllocator(std::string filename, int flags, size_t map_size=-1L)
    : filename(filename), flags(flags), size_(map_size) {}
    void close() {
        std::clog << "A CustomLibshmAllocator object to be destructed" << std::endl;
    }
    ~CustomLibshmAllocator() {
        this->close();
    }
    size_t size() const { return size_; }
    // creates new mmap/shm files on demand
    c10::DataPtr allocate(size_t alloc_size) const override {
        TORCH_CHECK(
            this->size() >= alloc_size, "trying to alloc more than reserved: ",
            alloc_size, " vs. ", this->size()
        );
        std::string fname = filename;
        auto data_ptr = THManagedMapAllocator::makeDataPtr(nullptr, filename.data(), flags, size_);
        return data_ptr;
    }
    static std::shared_ptr<Allocator>
    New(std::string filename, int flags, size_t size) {
        return std::make_shared<CustomLibshmAllocator>(filename, flags, size);
    }
};
*/

// TODO: AllocatorWrapper needs to survive across process forks
// - need to define __getstate__, __setstate__ on the exposed Python class
class AllocatorWrapper: public torch::CustomClassHolder {
  private:
    // Python can't manage objects of custom classes but a smart pointer of the class
    using PythonObj = c10::intrusive_ptr<AllocatorWrapper>;
    // we want to wrap the raw pointer to the allocator in a shared_ptr
    // so that the allocator object itself could get free automatically
    std::shared_ptr<Allocator> allocator_;
  public:
    AllocatorWrapper(std::shared_ptr<Allocator> allocator)
    : allocator_(allocator) {}
    AllocatorWrapper(std::intptr_t allocator)
    : AllocatorWrapper(
        StaticAllocatorWrapper::New(reinterpret_cast<Allocator *>(allocator))
    ) {}
    // python requires intptr_t for raw pointers
    std::intptr_t get() {
        return reinterpret_cast<std::intptr_t>(allocator_.get());
    }

  private:
    static inline int get_flags(bool ro, bool temporary, bool use_shm) {
        int flags = 0;
        if (!ro) {
            // mmap with MAP_SHARED
            if (use_shm) {
                flags |= at::MappedAllocatorModes::ALLOCATOR_MAPPED_SHAREDMEM;
                // linux kernel seems to automatically free the allocated shared memory under
                // /dev/shm, even if we don't unlink it manually here
            } else {
                flags |= at::MappedAllocatorModes::ALLOCATOR_MAPPED_SHARED;
            }
            if (temporary)
                flags |= at::MappedAllocatorModes::ALLOCATOR_MAPPED_UNLINK;
        } else {
            flags |= at::MappedAllocatorModes::ALLOCATOR_MAPPED_NOCREATE;
        }
        return flags;
    }
    static PythonObj get_allocator_from_raw(Allocator *raw) {
        return c10::make_intrusive<AllocatorWrapper>(
            reinterpret_cast<std::intptr_t>(raw)
        );
    }
  public:
    static PythonObj default_cpu_allocator() {
        return get_allocator_from_raw(c10::GetDefaultCPUAllocator());
    }
    static PythonObj mmap_allocator(
        std::string filename, int64_t pool_size, bool ro, bool temporary, bool use_shm=false
    ) {
        int flags = get_flags(ro, temporary, use_shm);
        flags |= at::MappedAllocatorModes::ALLOCATOR_MAPPED_KEEPFD;
        auto allocator = CustomMapAllocator::New(filename, flags, pool_size);
        return c10::make_intrusive<AllocatorWrapper>(allocator);
    }
    // static PythonObj libshm_allocator(std::string filename, int64_t pool_size) {
    //     int flags = get_flags(/*ro=*/false, /*temp=*/false, /*use_shm=*/true);
    //     auto allocator = CustomLibshmAllocator::New(filename, flags, pool_size);
    //     return c10::make_intrusive<AllocatorWrapper>(allocator);
    // }
};

TORCH_LIBRARY(xTensor, m) {
    // allocators
    m.class_<AllocatorWrapper>("AllocWrapper")
     .def(torch::init<std::intptr_t>())
     .def("get", &AllocatorWrapper::get);

    m.def(
        "default_allocator() -> __torch__.torch.classes.xTensor.AllocWrapper",
        &AllocatorWrapper::default_cpu_allocator
    );
    m.def(
        "mmap_allocator(str fname, int size, bool read_only, bool temp, bool shm)"
        "-> __torch__.torch.classes.xTensor.AllocWrapper",
        &AllocatorWrapper::mmap_allocator
    );
    /*
    m.def(
        "libshm_allocator(str fname, int size)"
        "-> __torch__.torch.classes.xTensor.AllocWrapper",
        &AllocatorWrapper::libshm_allocator
    );
    */

    // ops
    m.def(
        "scatter_index(Tensor out, Tensor index, Tensor intervals) -> Tensor out_ref",
        &scatter_index
    );
    m.def(
        "scatter_copy(Tensor out, Tensor index, Tensor src) -> Tensor out_ref",
        &scatter_copy
    );
    m.def(
        "ranges_gather(Tensor out, Tensor src, Tensor starts, Tensor sizes) -> Tensor",
        py::overload_cast<Tensor&, const Tensor &, const Tensor&, const Tensor&>(&ranges_gather)
    );
    m.def(
        "ranges_gather_io(Tensor out, str filename, Tensor starts, Tensor sizes) -> Tensor",
        py::overload_cast<Tensor&, const std::string&, const Tensor&, const Tensor&>(&ranges_gather)
    );
    m.def(
        "ranges_add(Tensor target, Tensor starts, Tensor sizes, Tensor values) -> Tensor",
        &ranges_add
    );
    m.def(
        "coo_list_merge(int num_nodes, (Tensor, Tensor)[] edges)"
        "-> (Tensor, Tensor)",
        &coo_list_merge
    );
    m.def(
        "coo_ranges_merge(int n, (Tensor, Tensor)[] edges, Tensor[] starts, Tensor[] sizes)"
        "-> (Tensor, Tensor)",
        &coo_ranges_merge
    );

    // utilities
    m.def(
        "check_madv_populate() -> bool",
        &check_madv_populate
    );
}

#ifndef GNNOS_TENSOR_IO_HPP_
#define GNNOS_TENSOR_IO_HPP_

#include <string>
#include <vector>
#include <memory>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <torch/torch.h>

namespace gnnos {

extern std::string TMPDIR;
extern int IO_THREADS;

class Store {
public:
    using Handle = std::shared_ptr<Store>;

    Store(const Store &) = delete;
    Store(Store &&) = delete;
    virtual ~Store();

    ssize_t pread(void *buf, size_t nbytes, off_t offset) const {
        return pread64(fd_, buf, nbytes, offset);
    }

    ssize_t pwrite(const void *buf, size_t nbytes, off_t offset) const {
        return pwrite64(fd_, buf, nbytes, offset);
    }

    int alloc(size_t offset, size_t len) {
        return posix_fallocate(fd_, offset, len);
    }

    long size() const;

    int persist(const std::string &path);

    const std::string &path() const { return path_; }

    static Handle Open(const char *path, int flags);
    static Handle OpenTemp(const char *path);
private:
    std::string path_;
    int fd_;
    bool is_tmp_;

    Store(const char *path, int fd, bool is_tmp): path_(path), fd_(fd), is_tmp_(is_tmp) {}
};

// TODO
class MmapStore {
public:
    using Handle = std::shared_ptr<MmapStore>;

    MmapStore(const MmapStore &) = delete;
    MmapStore(MmapStore &&) = delete;
    ~MmapStore();

    ssize_t pread(void *buf, size_t nbytes, off_t offset) const {
        memcpy(buf, ptr_ + offset, nbytes);
        return nbytes;
    }

    ssize_t pwrite(const void *buf, size_t nbytes, off_t offset) const {
        memcpy(ptr_ + offset, buf, nbytes);
        return nbytes;
    }

    int alloc(size_t offset, size_t len) {
        if (ptr_ != NULL)
            munmap(ptr_, length_);
        posix_fallocate(fd_, offset, len);
        ptr_ = (char *)mmap(0, length_, PROT_READ, MAP_SHARED, fd_, 0);
        return 0;
    }

    long size() const;

    int persist(const std::string &path);

    const std::string &path() const { return path_; }

    static Handle Open(const char *path, int flags);
private:
    std::string path_;
    int fd_ = -1;
    char *ptr_ = NULL;
    long length_;
    bool is_tmp_;

    MmapStore(const char *path, int fd, bool is_tmp): path_(path), fd_(fd), is_tmp_(is_tmp) {}
};

c10::ScalarType default_dtype(int itemsize);

template <typename T>
using SmallVector = c10::SmallVector<T, 4>;

class TensorInfo {
    std::string path_;
    SmallVector<long> shape_;
    int itemsize_ = 4;
    long offset_ = 0;
public:
    const std::string &path() const { return path_; }
    const c10::IntArrayRef shape() const { return shape_; }
    int itemsize() const { return itemsize_; }
    long offset() const { return offset_; }
    TensorInfo &path(std::string path) {
        path_ = std::move(path);
        return *this;
    }
    TensorInfo &shape(c10::IntArrayRef new_shape) {
        shape_ = new_shape;
        return *this;
    }
    TensorInfo &itemsize(int new_size) {
        itemsize_ = new_size;
        return *this;
    }
    TensorInfo &offset(long new_offset) {
        offset_ = new_offset;
        return *this;
    }
};
TensorInfo TensorOptions();
TensorInfo TensorOptions(std::string path);
std::ostream& operator<<(std::ostream&, const TensorInfo &);

class TensorStore {
public:
    // TODO: we shouldn't allow an empty constructor: hdl in an illegal state
    TensorStore() {}
    TensorStore(const TensorStore &store) = default;
    TensorStore(TensorStore &&store) = default;
    TensorStore& operator=(TensorStore &&store) = default;
    ~TensorStore() = default;

    // create from TensorInfo: ("path", "shape", "dtype", "offset")
    static TensorStore
    NewFrom(TensorInfo option, int flags);
    static TensorStore
    OpenForRead(TensorInfo option = TensorOptions());
    static TensorStore
    Open(TensorInfo option = TensorOptions());
    static TensorStore
    Create(TensorInfo option = TensorOptions());
    static TensorStore
    CreateTemp(TensorInfo option = TensorOptions());

    // TODO: we really need to replace itemsize with dtype
    // read TensorStore into a torch Tensor
    torch::Tensor tensor() const;
    torch::Tensor tensor(torch::ScalarType) const;
    // torch::Tensor &TensorStore::tensor_out(torch::Tensor &out) const;

    // copy the tensor to another
    void copy_to(TensorStore &);
    void save_to(std::string path) {
        TORCH_CHECK(hdl->persist(path) == 0, "Failed to save as ",
            path, ": ", strerror(errno));
    }

    // slice TensorStore[start, end) at dim=0
    TensorStore slice(long start, long end) const {
        return TensorStore(*this, std::make_pair(start, end));
    }
    // slice TensorStore[0, end) at dim=0
    TensorStore slice(long end) const { return this->slice(0, end); }
    // read TensorStore[idx] into a torch Tensor
    torch::Tensor at(long idx) const { return slice(idx, idx+1).tensor(); }
    // TODO: batched random read
    torch::Tensor gather(torch::Tensor indices /*, int dim=0*/) const;

    TensorStore &reshape(c10::IntArrayRef new_shape);
    TensorStore &flatten() { return this->reshape({numel()}); }

    // constant methods
    const c10::IntArrayRef shape() const { return shape_; }
    long numel() const { return size_; }
    int itemsize() const { return itemsize_; }
    TensorInfo metadata() const {
        return TensorOptions(hdl->path()).shape(shape_)
            .itemsize(itemsize_).offset(seek_set);
    }

    // store-flavored read/write
    inline ssize_t pread(void *buf, size_t nbytes, long offset) const {
        TORCH_CHECK(offset + (long)nbytes <= seek_set + size_ * itemsize_,
            "Store read out of bound: ", std::make_pair(offset, nbytes+offset));
        return hdl->pread(buf, nbytes, this->seek_set + offset);
    }
    inline ssize_t pwrite(const void *buf, size_t nbytes, long offset) const {
        TORCH_CHECK(offset + (long)nbytes <= seek_set + size_ * itemsize_,
            "Store write out of bound: ", std::make_pair(offset, nbytes+offset));
        return hdl->pwrite(buf, nbytes, this->seek_set + offset);
    }

    template <typename T>
    class Accessor {
    public:
        Accessor(const TensorStore &store): store_(store) {
            TORCH_CHECK(sizeof(T) == (size_t) store.itemsize_,
                "mismatched itemsize and sizeof(T)=", sizeof(T));
        }

        long size() const { return store_.numel(); }

        // accessor-flavored methods
        T operator[](size_t idx) const {
            T elem;
            TORCH_CHECK(store_.pread(&elem, sizeof(T), idx*sizeof(T))>=0,
                strerror(errno));
            return elem;
        }
        void put(const T &data, size_t idx) const {
            TORCH_CHECK(store_.pwrite(&data, sizeof(T), idx*sizeof(T))>=0,
                strerror(errno));
        }
        std::vector<T> slice(size_t start, size_t end) const {
            std::vector<T> array(end-start);
            TORCH_CHECK(store_.pread(&array[0], sizeof(T)*array.size(), start*sizeof(T))>=0,
                strerror(errno));
            return array;
        }
        void slice_put(const T *data, size_t start, size_t end) const {
            TORCH_CHECK(store_.pwrite(data, sizeof(T)*(end-start), start*sizeof(T))>=0,
                strerror(errno));
        }
    private:
        const TensorStore &store_;
    };

    // provide typed array-like access to the TensorStore
    // NB: allows modification to the underlying data even if *this is const
    template <typename T>
    Accessor<T> accessor() const& {
        return Accessor<T>(*this);
    }
    // prohibit creating accessors on temporary TensorStore obj
    template <typename T>
    Accessor<T> accessor() && = delete;

protected:
    Store::Handle hdl;
    // shape_ and size_ are counts of items
    SmallVector<long> shape_;
    long size_ = 0;
    // TODO: consider changing itemsize to torch::dtype
    int itemsize_ = 0;
    // seek_* are byte offsets
    long seek_set = 0;

    // create from another TensorStore by slicing dim-0 range
    TensorStore(const TensorStore &other, std::pair<long, long> range);
    // create from an existing store handle
    TensorStore(Store::Handle hdl, c10::IntArrayRef shape, int itemsize, long offset);
};

// Gather store slices into a torch Tensor
torch::Tensor GatherSlices(
    const TensorStore &store,
    const std::vector<std::pair<long, long>> &ranges,
    torch::ScalarType);

// Shuffle Store[shuffled[i]] -> Store[i]
void ShuffleStore(TensorStore &, const TensorStore &, const torch::Tensor &);

}   // namespace gnnos

#endif
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <sys/stat.h>

#include "tensor_io.hpp"
#include "utils.hpp"

namespace gnnos {

std::string TMPDIR = "/mnt/md0/tmp";
int IO_THREADS = torch::get_num_threads();

// Store methods

Store::~Store() {
    LOG(INFO) << "Close " << path_ << " (fd=" << fd_ << ")";
    if (!is_tmp_ && fsync(fd_) != 0)
        TORCH_WARN("Failed to fsync fd=", fd_, ": ", strerror(errno));
    if (is_tmp_ && unlink(path_.data()) != 0)
        TORCH_WARN("Failed to unlink fd=", fd_, ": ", strerror(errno));
    if (close(fd_) != 0)
        TORCH_WARN("Failed to close fd=", fd_, ": ", strerror(errno));
}

long Store::size() const {
    struct stat statbuf;
    TORCH_CHECK(fstat(this->fd_, &statbuf) >= 0,
        "Failed to stat " + path_ + ": ", strerror(errno));
    return statbuf.st_size;
}

Store &Store::persist(const std::string &path) {
    is_tmp_ = false;
    char proc_name[255];
    snprintf(proc_name, sizeof(proc_name), "/proc/self/fd/%d", fd_);
    int rc = linkat(AT_FDCWD, proc_name, AT_FDCWD, path.data(), AT_SYMLINK_FOLLOW);
    TORCH_CHECK(rc == 0, "Failed to persist Store (fd=", fd_, ") to ", path);
    TORCH_CHECK(unlink(this->path_.data()) == 0, "Failed to unlink ",
        this->path_, ": ", strerror(errno));
    this->path_ = path;
    return *this;
}

Store::Handle Store::Open(const char *path, int flags) {
    int mask = S_IRUSR | S_IRGRP | S_IROTH;
    if ((flags & O_WRONLY) | (flags & O_RDWR) ) mask |= S_IWUSR;
    int fd = open(path, flags, mask);
    TORCH_CHECK(fd >= 0, "Failed to open Store from ", path, ": ", strerror(errno));
    LOG(INFO) << "Open " << path << " (fd=" << fd << ")";
    bool is_tmp = (flags & O_TMPFILE) != 0;
    return std::shared_ptr<Store>(new Store(path, fd, is_tmp));
}

Store::Handle Store::OpenTemp(const char *path) {
    char filename[256];
    strncpy(filename, path, 100);
    strncat(filename, "/gnnos-XXXXXX", 100);
    int fd = mkstemp(filename);
    TORCH_CHECK(fd >= 0, "Failed to open temp Store from ", path, ": ", strerror(errno));
    LOG(INFO) << "Open TempStore (fd=" << fd << ")";
    return std::shared_ptr<Store>(new Store(filename, fd, true));
}


// MmapStore methods

MmapStore::~MmapStore() {
    if (munmap(ptr_, length_) != 0)
        TORCH_WARN("Failed to munmap fd=", fd_, ": ", strerror(errno));
    if (!is_tmp_ && fsync(fd_) != 0)
        TORCH_WARN("Failed to fsync fd=", fd_, ": ", strerror(errno));
    if (close(fd_) != 0)
        TORCH_WARN("Failed to close fd=", fd_, ": ", strerror(errno));
    TORCH_WARN("Close ", path_ , " (fd=", fd_, ")");
}

MmapStore::Handle MmapStore::Open(const char *path, int flags) {
    int mask = S_IRUSR | S_IRGRP | S_IROTH;
    if (flags & O_WRONLY) mask |= S_IWUSR;
    int fd = open(path, flags, mask);
    TORCH_CHECK(fd >= 0, "Failed to open Store from ", path);
    LOG(INFO) << "Open " << path << " (fd=" << fd << ")";
    bool is_tmp = (flags & O_TMPFILE) != 0;
    return std::shared_ptr<MmapStore>(new MmapStore(path, fd, is_tmp));
}


TensorInfo TensorOptions() { return TensorInfo(); }
TensorInfo TensorOptions(std::string path) { return TensorInfo().path(std::move(path)); }
std::ostream& operator<<(std::ostream& out, const TensorInfo &info) {
    out << "TensorInfo {";
    out << '\"' << info.path() << '\"' <<  ", ";
    out << info.shape() << ", ";
    out << "dtype:" << info.dtype() << ", ";
    out << "offset:" << info.offset();
    out << "}";
    return out;
}

// TensorStore methods
TensorStore::TensorStore(Store::Handle hdl, c10::IntArrayRef shape,
                         DType dtype, long offset)
    : hdl(std::move(hdl)), shape_(std::move(shape)), dtype_(dtype), seek_set(offset)
{
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1L, std::multiplies<long>());
};

TensorStore::TensorStore(const TensorStore &other, std::pair<long, long> range)
    : TensorStore(other)
{
    // sanity check
    TORCH_CHECK(range.first <= range.second, "Invalid slice arguments ", range);
    TORCH_CHECK(range.second <= shape_[0], "Invalid slice arguments ", range);

    auto row_items = other.numel() / other.shape_[0];
    shape_[0] = range.second - range.first;
    size_ = row_items * shape_[0];
    seek_set = other.seek_set + row_items * itemsize() * range.first;
}

TensorStore TensorStore::NewFrom(TensorInfo option, int flags) {
    auto tensor = TensorStore(Store::Open(option.path().data(), flags),
        option.shape(), option.dtype(), option.offset());
    long len = tensor.numel() * tensor.itemsize();
    if ((flags & O_WRONLY) | (flags & O_RDWR) ) {
        int rc = tensor.hdl->alloc(tensor.seek_set, len);
        TORCH_CHECK(rc == 0, "Failed to alloc store: ", strerror(rc));
    }
    return tensor;
}
TensorStore TensorStore::OpenForRead(TensorInfo option) {
    return NewFrom(option, O_RDONLY);
}
TensorStore TensorStore::Open(TensorInfo option) {
    return NewFrom(option, O_RDWR);
}
TensorStore TensorStore::Create(TensorInfo option) {
    return NewFrom(option, O_CREAT | O_RDWR | O_EXCL);
}
TensorStore TensorStore::CreateTemp(TensorInfo option) {
    auto tensor = TensorStore(Store::OpenTemp(option.path().data()),
        option.shape(), option.dtype(), option.offset());
    long len = tensor.numel() * tensor.itemsize();
    int rc = tensor.hdl->alloc(tensor.seek_set, len);
    TORCH_CHECK(rc == 0, "Failed to alloc store: ", strerror(rc));
    return tensor;
}

TensorStore TensorStore::NewFrom(const torch::Tensor &tensor) {
    auto option = TensorOptions(TMPDIR).shape(tensor.sizes()).dtype(tensor.scalar_type());
    auto store = CreateTemp(option);
    auto data_ptr = tensor.data_ptr();
    store.pwrite(data_ptr, tensor.nbytes(), 0);
    return store;
}
TensorStore TensorStore::NewFrom(const torch::Tensor &tensor, std::string path) {
    auto option = TensorOptions(path).shape(tensor.sizes()).dtype(tensor.scalar_type());
    auto store = Create(option);
    auto data_ptr = tensor.data_ptr();
    store.pwrite(data_ptr, tensor.nbytes(), 0);
    return store;
}

torch::Tensor TensorStore::tensor() const {
    return tensor(this->dtype());
}
torch::Tensor TensorStore::tensor(torch::ScalarType dtype) const {
    TORCH_CHECK(dtype_ == dtype || torch::elementSize(dtype_) == torch::elementSize(dtype),
        "Illegal dtype cast from ", dtype_, " to ", dtype);
    size_t sz = this->numel() * this->itemsize();
    char *buf = new char[sz];
    this->pread(buf, sz, 0);
    return torch::from_blob(
        buf, this->shape(),
        [](void *buf) { delete[] (char *)buf; },
        torch::dtype(dtype));
}

void TensorStore::copy_to(TensorStore &that) const {
    TORCH_CHECK(torch::elementSize(dtype_) == torch::elementSize(that.dtype_),
        "Illegal dtype cast from ", dtype_, " to ", that.dtype_);
    constexpr size_t CHUNK_SIZE = 1024 * 1024;
    size_t tensor_nbytes = this->numel() * this->itemsize();
    size_t nchunks = tensor_nbytes / CHUNK_SIZE;
    char buf[CHUNK_SIZE];
    for (size_t i = 0; i < nchunks; ++i) {
        pread(buf, CHUNK_SIZE, i * CHUNK_SIZE);
        that.pwrite(buf, CHUNK_SIZE, i * CHUNK_SIZE);
    }
    size_t remains = tensor_nbytes - CHUNK_SIZE * nchunks;
    if (remains > 0) {
        pread(buf, remains, CHUNK_SIZE * nchunks);
        that.pwrite(buf, remains, CHUNK_SIZE * nchunks);
    }
}

TensorInfo TensorStore::save_to(std::string path) const {
    if (hdl->path() == path) {
        // do nothing if copy to itself
        return metadata();
    }
    auto option = metadata().path(path);
    if (hdl->is_tmp()) {
        hdl->persist(path);
    } else {
        auto new_store = TensorStore::Create(option);
        this->copy_to(new_store);
    }
    return option;
}

TensorStore &TensorStore::reshape(c10::IntArrayRef new_shape) {
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(),
        1L, std::multiplies<long>());
    TORCH_CHECK(new_size == numel(), "Invalid reshape to ", c10::IntArrayRef(new_shape));
    this->shape_ = new_shape;
    return *this;
}

ssize_t TensorStore::pread(void *buf, size_t nbytes, long offset) const {
    TORCH_CHECK(offset + (long)nbytes <= seek_set + size_ * itemsize(),
        "Store read out of bound: ", std::make_pair(offset, nbytes+offset));
    ssize_t bytes = 0;
    do {
        auto read_bytes = hdl->pread((char *)buf + bytes, nbytes - bytes,
            seek_set + offset + bytes);
        TORCH_CHECK(read_bytes > 0, "Store read terminated early at ",
            bytes, "(requested ", nbytes, ")");
        bytes += read_bytes;
    } while (bytes < nbytes);
    return bytes;
}

ssize_t TensorStore::pwrite(const void *buf, size_t nbytes, long offset) const {
    TORCH_CHECK(offset + (long)nbytes <= seek_set + size_ * itemsize(),
        "Store write out of bound: ", std::make_pair(offset, nbytes+offset));
    ssize_t bytes = 0;
    do {
        auto write_bytes = hdl->pwrite((char *)buf + bytes, nbytes - bytes,
            seek_set + offset + bytes);
        TORCH_CHECK(write_bytes > 0, "Store write terminated early at ",
            bytes, "(requested ", nbytes, ")");
        bytes += write_bytes;
    } while (bytes < nbytes);
    return bytes;
}

torch::Tensor GatherSlices(
    const TensorStore &store,
    const std::vector<std::pair<long, long>> &ranges)
{
    std::vector<TensorStore> store_slices;
    long rows = 0;
    std::vector<long> pos = {0};
    for (auto r : ranges) {
        store_slices.push_back(store.slice(r.first, r.second));
        rows += r.second - r.first;
        pos.push_back(rows);
    }
    SmallVector<long> shape(store.shape());
    long row_numel = store.numel() / shape[0];
    shape[0] = rows;

    char *buf = new char[rows * row_numel * store.itemsize()];

    #pragma omp parallel for num_threads(IO_THREADS/2)
    for (size_t i = 0; i < ranges.size(); ++i) {
        long start = pos[i] * row_numel * store.itemsize();
        long end = pos[i+1] * row_numel * store.itemsize();
        store_slices[i].pread(buf + start,  end - start, 0);
    }

    // auto dtype = default_dtype(store.itemsize());
    return torch::from_blob(
        buf, shape,
        [](void *buf) { delete[] (char *)buf; },
        store.dtype());
}

void ShuffleStore(TensorStore &store, const TensorStore &from_store,
    const torch::Tensor &shuffled_ids) {

    auto id_accessor = shuffled_ids.accessor<long, 1>();
    constexpr long BLOCK_SZ = 1024 * 64;
    SmallVector<long> buf_shape(from_store.shape());
    buf_shape[0] = BLOCK_SZ;

    long offset = 0;
    auto tensor_buf = torch::empty(buf_shape, torch::dtype(from_store.dtype()));
    char *buf = (char *)tensor_buf.data_ptr();
    const long row_bytes = tensor_buf.nbytes() / BLOCK_SZ;

    for(long i = 0; i < id_accessor.size(0); i += BLOCK_SZ) {
        long start = i;
        long end = i + BLOCK_SZ;
        if (end > id_accessor.size(0)) end = id_accessor.size(0);
        #pragma omp parallel for num_threads(IO_THREADS)
        for (long j = start; j < end; ++j) {
            long from_id = id_accessor[j];
            from_store.pread(buf + (j - start) * row_bytes, row_bytes, from_id * row_bytes);
        }
        store.pwrite(buf, (end - start) * row_bytes, offset);
        offset += (end - start) * row_bytes;
    }
}


}

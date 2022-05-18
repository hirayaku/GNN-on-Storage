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

int Store::persist(const std::string &path) {
    this->path_ = path;
    if (is_tmp_) {
        is_tmp_ = false;
        char proc_name[255];
        snprintf(proc_name, sizeof(proc_name), "/proc/self/fd/%d", fd_);
        return linkat(AT_FDCWD, proc_name, AT_FDCWD, path.data(), AT_SYMLINK_FOLLOW);
    } else {
        return linkat(AT_FDCWD, path_.data(), AT_FDCWD, path.data(), 0);
    }
}

Store::Handle Store::Open(const char *path, int flags) {
    int mask = S_IRUSR | S_IRGRP | S_IROTH;
    if (flags & O_WRONLY) mask |= S_IWUSR;
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
    LOG(INFO) << "Open " << filename << " (fd=" << fd << ")";
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



c10::ScalarType default_dtype(int itemsize) {
    auto dtype = torch::kLong;
    switch(itemsize) {
    case 1:
        dtype = torch::kByte;
        break;
    case 2:
        dtype = torch::kInt16;
        break;
    case 4:
        dtype = torch::kInt32;
        break;
    case 8:
        dtype = torch::kInt64;
        break;
    default:
        throw std::runtime_error(
            "Can't find dtype for itemsize " + std::to_string(itemsize)
        );
    }
    return dtype;
}


// TensorInfo methods
TensorInfo TensorOptions() { return TensorInfo(); }
TensorInfo TensorOptions(std::string path) { return TensorInfo().path(std::move(path)); }
std::ostream& operator<<(std::ostream& out, const TensorInfo &info) {
    out << "TensorInfo{";
    out << '\"' << info.path() << '\"' <<  ", ";
    out << info.shape() << ", ";
    out << "itemsize:" << info.itemsize() << ", ";
    out << "offset:" << info.offset();
    out << "}";
    return out;
}

// TensorStore methods
TensorStore::TensorStore(Store::Handle hdl, c10::IntArrayRef shape,
                         int itemsize, long offset)
    : hdl(std::move(hdl)), shape_(std::move(shape)), itemsize_(itemsize), seek_set(offset)
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
    seek_set = other.seek_set + row_items * itemsize_ * range.first;
}

TensorStore TensorStore::NewFrom(TensorInfo option, int flags) {
    auto tensor = TensorStore(Store::Open(option.path().data(), flags),
        option.shape(), option.itemsize(), option.offset());
    long len = tensor.numel() * tensor.itemsize();
    if (tensor.hdl->size() < tensor.seek_set + len) {
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
        option.shape(), option.itemsize(), option.offset());
    long len = tensor.numel() * tensor.itemsize();
    if (tensor.hdl->size() < tensor.seek_set + len) {
        int rc = tensor.hdl->alloc(tensor.seek_set, len);
        TORCH_CHECK(rc == 0, "Failed to alloc store: ", strerror(rc));
    }
    return tensor;
    // return NewFrom(option, O_TMPFILE | O_RDWR);
}

torch::Tensor TensorStore::tensor() const {
    auto dtype = default_dtype(this->itemsize());
    return tensor(dtype);
}
torch::Tensor TensorStore::tensor(torch::ScalarType dtype) const {
    size_t sz = this->numel() * this->itemsize();
    char *buf = new char[sz];
    this->pread(buf, sz, 0);
    return torch::from_blob(
        buf, this->shape(),
        [](void *buf) { delete[] (char *)buf; },
        torch::dtype(dtype));
}

void TensorStore::copy_to(TensorStore &that) {
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

TensorStore &TensorStore::reshape(c10::IntArrayRef new_shape) {
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(),
        1L, std::multiplies<long>());
    TORCH_CHECK(new_size == numel(), "Invalid reshape to ", c10::IntArrayRef(new_shape));
    this->shape_ = new_shape;
    return *this;
}

torch::Tensor GatherSlices(
    const TensorStore &store,
    const std::vector<std::pair<long, long>> &ranges,
    torch::ScalarType dtype)
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
        torch::dtype(dtype));
}

void ShuffleStore(TensorStore &store, const TensorStore &from_store,
    const torch::Tensor &shuffled_ids) {

    auto id_accessor = shuffled_ids.accessor<long, 1>();
    constexpr long BLOCK_SZ = 1024 * 16;
    SmallVector<long> buf_shape(from_store.shape());
    buf_shape[0] = BLOCK_SZ;

    long offset = 0;
    auto tensor_buf = torch::empty(
        buf_shape,
        torch::dtype(default_dtype(from_store.itemsize()))
    );
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

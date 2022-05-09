#include <iomanip>
#include <numeric>
#include <sys/stat.h>

#include "tensor_io.hpp"

namespace gnnos {

long Store::size() const {
    struct stat statbuf;
    CHECK_GE(fstat(this->fd_, &statbuf), 0) << "Fail to stat " << path_ << strerror(errno);
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
    // using permission mask 0644 by default
    int fd = open(path, flags, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    CHECK_GE(fd, 0) << "Fail to open " << path << ": " << strerror(errno);
    LOG(WARNING) << "Open Store at " << path << "(fd=" << fd << ")";
    bool is_tmp = (flags & O_TMPFILE) != 0;
    return std::shared_ptr<Store>(new Store(path, fd, is_tmp));
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
TensorStore::TensorStore(Store::Handle hdl, SmallVector<size_t> shape,
                         size_t itemsize, size_t offset)
    : hdl(std::move(hdl)), shape_(std::move(shape)), itemsize_(itemsize), seek_set(offset)
{
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
};

TensorStore::TensorStore(const TensorStore &other, std::pair<size_t, size_t> range)
    : TensorStore(other)
{
    // sanity check
    CHECK_LE(range.first, range.second);
    CHECK_LE(range.second, shape_[0]);

    shape_[0] = range.second - range.first;
    size_ = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
    auto size_dim0 = size_ / shape_[0];
    seek_set = other.seek_set + size_dim0 * itemsize_ * range.first;
}

TensorStore TensorStore::NewFrom(TensorInfo option, int flags) {
    auto tensor = TensorStore(Store::Open(option.path().data(), flags),
        option.shape(), option.itemsize(), option.offset());
    size_t len = tensor.numel() * tensor.itemsize();
    if (tensor.hdl->size() < tensor.seek_set + len) {
        int rc = tensor.hdl->alloc(tensor.seek_set, len);
        CHECK_EQ(rc, 0) << strerror(rc);
    }
    return std::move(tensor);
}
TensorStore TensorStore::OpenForRead(TensorInfo option) {
    return NewFrom(option, O_RDONLY);
}
TensorStore TensorStore::Open(TensorInfo option) {
    return NewFrom(option, O_RDWR);
}
TensorStore TensorStore::Create(TensorInfo option) {
    return NewFrom(option, O_CREAT | O_RDWR);
}
TensorStore TensorStore::CreateTemp(TensorInfo option) {
    return NewFrom(option, O_TMPFILE | O_RDWR);
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

TensorStore &TensorStore::reshape(const SmallVector<size_t> &new_shape) {
    auto new_size = std::accumulate(new_shape.begin(), new_shape.end(),
        1UL, std::multiplies<size_t>());
    CHECK_EQ(new_size, numel()) << "TensorStore::reshape: can't reshape to "
        << c10::ArrayRef<size_t>(new_shape) << "";
    this->shape_ = new_shape;
    return *this;
}

}

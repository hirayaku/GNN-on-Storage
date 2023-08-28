#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <map>
#include <mutex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// static void init_msghdr(
//     struct msghdr *msg, struct iovec *iov, char *buf, size_t buf_len
// ) {
//     iov->iov_base = buf;
//     iov->iov_len = 1;
//     msg->msg_iov = iov;
//     msg->msg_iovlen = 1;
//     msg->msg_control = buf;
//     msg->msg_controllen = buf_len;
//     msg->msg_name = NULL;
//     msg->msg_namelen = 0;
// }

// // receive just one fd from socket
// static int recv_fd(int sock_fd) {
//     msghdr msg;
//     iovec iov;
//     char msg_buf[CMSG_SPACE(sizeof(int))];
//     init_msghdr(&msg, &iov, msg_buf, sizeof(msg_buf));

//     if (recvmsg(sock_fd, &msg, 0) == -1) {
//         return -1;
//     }

//     int found_fd = -1;
//     for (struct cmsghdr *header = CMSG_FIRSTHDR(&msg); header != NULL; header = CMSG_NXTHDR(&msg, header)) {
//         if (header->cmsg_level == SOL_SOCKET && header->cmsg_type == SCM_RIGHTS) {
//             int count = (header->cmsg_len - (CMSG_DATA(header) - (unsigned char *)header)) / sizeof(int);
//             for (int i = 0; i < count; ++i) {
//                 if (found_fd == -1)
//                     found_fd = ((int *)CMSG_DATA(header))[i];
//             }
//         }
//     }

//     if (found_fd == -1)
//         errno = ENOENT;
//     return found_fd;
// }

// // connect to the memory pool server through socket, and mmap the memory pool
// // into the current address space
// MemoryPoolClient::MemoryPoolClient(const std::string &socket_file) {
//     this->sock_ = socket_file;

//     int sock_fd = socket(AF_LOCAL, SOCK_SEQPACKET, 0);
//     if (sock_fd == -1) goto error;
//     this->sock_fd_ = sock_fd;

//     // connect to server via socket
//     sockaddr_un addr;
//     addr.sun_family = AF_LOCAL;
//     strncpy(addr.sun_path, socket_file.data(), sizeof(addr.sun_path));
//     addr.sun_path[sizeof(addr.sun_path)-1] = '\0';
//     if (connect(sock_fd, (sockaddr *)&addr, sizeof(addr)) == -1) {
//         goto error;
//     }
//     this->sock_conn_ = true;

//     // wait for the messages containing fd and memory pool size
//     if ((this->mem_fd_ = recv_fd(sock_fd)) < 0) goto error;

// error:
//     std::ostringstream oss;
//     if (this->sock_fd_ == -1) {
//         oss << "Failed to create socket with path: " << socket_file;
//         throw std::runtime_error(oss.str());
//     }
//     close(this->sock_fd_);
//     if (!this->sock_conn_) {
//         oss << "Failed to connect to server socket: " << socket_file;
//         throw std::runtime_error(oss.str());
//     }
//     if (this->mem_fd_ == -1) {
//         oss << "Failed to retrieve fd from server";
//         throw std::runtime_error(oss.str());
//     }
//     close(this->mem_fd_);
//     if (this->ptr_ == MAP_FAILED) {
//         oss << "Failed to mmap from server provided fd";
//         throw std::runtime_error(oss.str());
//     }
// }

namespace py = pybind11;

#include <unistd.h>
#include <sys/mman.h>

class FlatBuffer {
public:
    FlatBuffer(int fd, py::ssize_t size, const std::string &session)
    : size_(size), session_(session) {
        void *buf = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
        if (buf == MAP_FAILED) {
            throw std::runtime_error("FlatBuffer: failed to mmap from fd");
        }
        this->data_ = buf;
        this->root_ = true;
    }
    // FlatBuffer is not copiable - it uniquely owns the underlying buffer
    FlatBuffer(const FlatBuffer &other) = delete;
    FlatBuffer(FlatBuffer &other, py::ssize_t offset, py::ssize_t size, const std::string &id)
    : data_(other.data_), offset_(offset), size_(size), id_(id), session_(other.session_) {
        this->client_ = exec_pyfunc("get_client", py::make_tuple(session_));
    }
    ~FlatBuffer() {
        try {
            this->release();
        } catch (const std::exception &e) {
            std::cerr << "Exception thrown during buffer release: " << e.what() << std::endl;
        }
    }

    bool is_root() const { return root_; }
    void *data() const { return (char *)data_ + offset_; }
    py::ssize_t offset() const { return offset_; }
    std::vector<py::ssize_t> size() const { return {size_}; }
    std::vector<py::ssize_t> strides() const { return {sizeof(char)}; }
    const std::string &id() const { return id_; }
    const std::string &session() const { return session_; }
    py::object client() const { return client_; }

    static py::object exec_pyfunc(const std::string &name, py::args args) {
        auto iter = callbacks_.find(name);
        if (iter == callbacks_.end()) {
            std::ostringstream oss;
            oss << "FlatBuffer: callback \"" << name << "\" not registered";
            throw std::runtime_error(oss.str());
        }
        const auto &fn = iter->second;
        return fn(*args);
    }
    static bool add_pyfunc(const std::string &name, const py::function &fn) {
        auto p = callbacks_.insert({name, fn});
        return p.second;
    }
    static bool del_pyfunc(const std::string &name) {
        auto erased = callbacks_.erase(name);
        return erased != 0;
    }
private:
    void release() {
        if (!this->root_) {
            // contact client to release non-root buffer
            exec_pyfunc("free_buffer", py::make_tuple(session_, id_));
            // std::clog << "Non-Root FlatBuffer destructed: " << size_ << std::endl;
        } else {
            munmap(this->data_, this->size_);
            // std::clog << "Root FlatBuffer destructed: " << size_ << std::endl;
        }
    }
private:
    bool root_ = false;
    void * data_ = nullptr;
    py::ssize_t offset_ = 0;
    py::ssize_t size_;
    const std::string id_;
    const std::string session_;
    py::object client_ = py::none();
    static std::map<std::string, py::function> callbacks_;
    static std::mutex lock_;
};

std::map<std::string, py::function> FlatBuffer::callbacks_{};

class ManagedBuffer {
public:
    ManagedBuffer(py::ssize_t size, const std::string &name): size_(size) {
        std::string mem_name = name;
        if (mem_name.size() == 0) { mem_name = "mm_buffer"; }
        int memfd = memfd_create(mem_name.data(), 0);
        if (memfd < 0)
            throw std::runtime_error("ManagedBuffer: failed to create memfd");
        if (ftruncate(memfd, size) < 0) {
            throw std::runtime_error("ManagedBuffer: failed to ftruncate memfd");
        }
        void *buf = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, memfd, 0);
        if (buf == MAP_FAILED) {
            throw std::runtime_error("ManagedBuffer: failed to mmap memfd");
        }
        this->memfd_ = memfd;
        this->buffer_ = buf;
    }
    ~ManagedBuffer() {
        std::clog << "ManagedBuffer: destructed: " << size_ << std::endl;
        release();
    }
    std::tuple<std::string, py::ssize_t> allocate(py::ssize_t size, py::ssize_t align);
    std::tuple<py::ssize_t, py::ssize_t> refer(const std::string &id);
    void free(const std::string &id);
    int memfd() const { return memfd_; }
private:
    std::string gen_id();
    void *mem_alloc(py::ssize_t size, py::ssize_t align);
    void mem_free(void *addr);
    int inc_ref(const std::string &id);
    int dec_ref(const std::string &id);
    void release() {
        if (refcounts_.size() > 1)
            std::cerr << "ManagedBuffer: not all allocations are freed" << std::endl;
        // TODO: clear malloc states
        munmap(this->buffer_, this->size_);
    }
private:
    int memfd_ = 0;
    void *buffer_ = nullptr;
    py::ssize_t size_ = 0;
    std::unordered_map<std::string, int> refcounts_;
    std::mutex lock_;
};

PYBIND11_MODULE(SharedBuffer, m) {
py::class_<FlatBuffer>(m, "FlatBuffer", py::buffer_protocol())
    .def(py::init<int, py::ssize_t, py::object>())
    /// method to create non-root buffers
    .def("slice", [](FlatBuffer &b, py::ssize_t offset, py::ssize_t size, const std::string &id) {
        return std::make_unique<FlatBuffer>(b, offset, size, id);
    })
    .def("is_root", &FlatBuffer::is_root)
    .def("data_ptr", &FlatBuffer::data)
    .def("size", &FlatBuffer::size)
    .def_property_readonly("id", &FlatBuffer::id)
    .def_property_readonly("session", &FlatBuffer::session)
    .def_static("register", &FlatBuffer::add_pyfunc)
    .def_static("deregister", &FlatBuffer::del_pyfunc)
    .def_static("call", &FlatBuffer::exec_pyfunc, py::arg("args")=py::make_tuple())
    /// pickling support
    .def(py::pickle(
        [](const FlatBuffer &b) {
            return FlatBuffer::exec_pyfunc("serialize", py::make_tuple(b));
        },
        [](py::tuple t) {
            auto obj = FlatBuffer::exec_pyfunc("deserialize", py::make_tuple(t));
            return obj.cast<FlatBuffer>();
        }
    ))
    /// Provide buffer access
    .def_buffer([](FlatBuffer &m) -> py::buffer_info {
        return py::buffer_info(
            (char *)m.data(),       /* Pointer to buffer */
            m.size(),               /* Buffer dimensions */
            m.strides()             /* Strides (in bytes) for each index */
        );
    });
}

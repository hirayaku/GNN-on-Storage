#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>

// ----------------
// Regular C++ code
// ----------------

template <typename T>
class local_buf {
  static const size_t FLUSH_CUTOFF = 128;
  const size_t feat_dim;
  std::vector<T> flattened;
  void *dst;

  public:
  local_buf(const size_t feat_dim, void *offset): feat_dim(feat_dim) {
    dst = offset;
    flattened.reserve(FLUSH_CUTOFF * feat_dim);
  }
  ~local_buf() { flush(); }

  const size_t num_feats() const {
    return flattened.size() / feat_dim;
  }
  void append(const T *begin, const T *end) {
    assert((end - begin) % feat_dim == 0 && "feature vector to append is not aligned");
    flattened.insert(flattened.end(), begin, end);
    if (flattened.size() >= FLUSH_CUTOFF * feat_dim) flush();
  }
  private:
  void flush() {
    std::memcpy(dst, flattened.data(), flattened.size() * sizeof(T));
    dst = (T *)dst + flattened.size();
    flattened.clear();
  }
};

void do_shuffle(const std::vector<long> &assignments, const std::vector<long> &offsets,
                const size_t feat_dim, float *in, float *out)
{
  std::vector<local_buf<float>> write_buffers;
  for (size_t i = 0; i < offsets.size(); ++i) {
    write_buffers.push_back(local_buf<float>(feat_dim, out + offsets[i] * feat_dim));
  }

  for (long nid = 0; nid <= assignments.size(); ++nid) {
    write_buffers[assignments[nid]].append(in + feat_dim*nid, in + feat_dim*(nid+1));
  }
}

// @param
// assignments:   std::vector ([...]) (read only) cluster assignments of each node
// num_clusters:  number of clusters
// feat_dim:      dimension of each node feature (1D vector)
// feats_in, feats_out: input and output feature files
void shuffle(const std::vector<long>& assignments, const std::vector<long> &clusters_cdf, const size_t feat_dim,
             const std::string feats_in, const std::string feats_out)
{
  const size_t feats_size = assignments.size() * feat_dim * sizeof(float);

  int fd_in = open(feats_in.data(), O_RDONLY);
  if (fd_in == -1) {
    perror("open feats_in");
    exit(errno);
  }

  struct stat feats_info;
  if (fstat(fd_in, &feats_info) != 0) {
    perror("stat");
    exit(errno);
  }
  assert(feats_info.st_size != feats_size && "input feature file size different from calculated size");
  
  void *fmap_in = mmap(0, feats_size, PROT_READ, MAP_SHARED, fd_in, 0);
  if (fmap_in == MAP_FAILED) {
    perror("mmap feats_in");
    exit(errno);
  }

  int fd_out = open(feats_out.data(), O_RDWR | O_CREAT, (mode_t)0664);
  if (fd_out == -1) {
    perror("open");
    exit(errno);
  }
  int rc = posix_fallocate(fd_out, 0, feats_size);
  if (rc != 0) {
    std::cerr << "posix_fallocate: " << rc << "\n";
    exit(rc);
  }
  void *fmap_out = mmap(0, feats_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);
  if (fmap_out == MAP_FAILED) {
    perror("mmap feats_out");
    exit(errno);
  }

  do_shuffle(assignments, clusters_cdf, feat_dim, (float *)fmap_in, (float *)fmap_out);

  munmap(fmap_out, feats_size);
  close(fd_out);
  munmap(fmap_in, feats_size);
  close(fd_in);
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(feat_shuffle,m)
{
  m.doc() = "Shuffle feature data based on cluster assignments";

  m.def("shuffle", &shuffle, "Shuffle feature data stored on the disk file based on cluster assignments");
  m.def("do_shuffle", &do_shuffle, "Shuffle feature data based on cluster assignments");
}

#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <omp.h>
#include "utils.hpp"
#include "ext_ops.hpp"

#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>

bool has_madv_populate = false;
static int _MADV_POPULATE_READ = 22;
static int _MADV_POPULATE_WRITE = 23;

bool check_madv_populate() {
    bool _has_madv_populate = false;
    int flag = _MADV_POPULATE_READ;
    void *map = nullptr;
    char tempfile[128] = {'\0'};
    const char *homedir = getenv("HOME");
    if (homedir) strncat(tempfile, homedir, 64);
    else tempfile[0] = '.';
    strncat(tempfile, "/madv_test_XXXXXX", 64);
    int fd = mkstemp(tempfile);
    if (fd < 0) {
        perror("mkstemp");
        return false;
    }
    unlink(tempfile);
    if (ftruncate(fd, 4096) != 0) {
        perror("ftruncate");
        goto cleanup;
    }
    map = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        goto cleanup;
    }
    if (madvise(map, 4096, flag) < 0) {
        _has_madv_populate = false;
    } else {
        _has_madv_populate = true;
    }
  cleanup:
    close(fd);
    has_madv_populate = _has_madv_populate;
    return _has_madv_populate;
}

static inline int batch_populate_pte(void *data_ptr, size_t size, bool write=false) {
    return madvise(data_ptr, size, write ? _MADV_POPULATE_WRITE : _MADV_POPULATE_READ);
}

// get the position that each element from the index array would be in the destination array
// it could be achieved with two passes of argsort, but we don't really need sorting here
// in order to maintain the stability, we use a sequential implementation
// NOTE: we could parallelize it with per-partition counting + prefix sum
Tensor &scatter_index(
    Tensor &out, const Tensor &index, const Tensor &intervals
) {
    auto start_pos = tensor_to_vector<long>(intervals);
    long *out_p = out.data_ptr<long>();
    const size_t size = index.size(0);

    AT_DISPATCH_INDEX_TYPES(
        index.scalar_type(), "ext_scatter_index", [&]() {
            const auto *index_p = index.data_ptr<index_t>();
            for (size_t i = 0; i < size; ++i) {
                auto slot = index_p[i];
                *out_p++ = start_pos[slot]++;
            }
        }
    );

    return out;
}

Tensor &scatter_copy(
    Tensor &out, const Tensor &index, const Tensor &src
) {
    // NOTE: built-in scatter doesn't parallelize well
    // if (src.dim() == 1) {
    //     return out.scatter_(0, index, src);
    // }

    TORCH_CHECK(index.scalar_type() == at::kLong, "index should have dtype long/int64_t");
    TORCH_CHECK(out.dim() == src.dim(), "out doesn't have the same dimension as src");
    TORCH_CHECK(out.scalar_type() == src.scalar_type());
    TORCH_CHECK(out.stride(0) == src.stride(0));

    const auto *index_p = index.data_ptr<long>();
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::Bool, src.scalar_type(), "ext_scatter_dim0", [&]() {
            const auto *src_p = src.data_ptr<scalar_t>();
            auto *out_p = out.data_ptr<scalar_t>();
            auto stride = out.stride(0);
            // torch::parallel_for (static scheduling) could lead to thrashing
            dynamic_parallel_for(0, index.size(0),
                [&](long i) {
                    long o = index_p[i];
                    std::copy(src_p + i * stride, src_p + (i+1) * stride, out_p + o * stride);
                }, 1024
            );
        }
    );

    return out;
}

Tensor &ranges_gather(
    Tensor &out, const Tensor &src, const Tensor &starts, const Tensor &lengths
) {
    TORCH_CHECK(starts.scalar_type() == at::kLong, "index should have dtype long/int64");
    TORCH_CHECK(lengths.scalar_type() == at::kLong, "sizes should have dtype long/int64");
    TORCH_CHECK(out.scalar_type() == src.scalar_type());
    TORCH_CHECK(out.stride(0) == src.stride(0));

    auto stride = src.stride(0);
    const auto *starts_p = starts.data_ptr<long>();
    const auto *lengths_p = lengths.data_ptr<long>();
    std::vector<long> offsets(lengths.size(0), 0);
    for (size_t i = 1; i < lengths.size(0); ++i) {
        offsets[i] += offsets[i-1] + lengths_p[i-1];
    }

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::Bool, src.scalar_type(), "ranges_gather_dim0", [&]() {
        const auto *src_p = src.data_ptr<scalar_t>();
        auto *out_p = out.data_ptr<scalar_t>();

        // if ranges are too small on average, don't populate ptes in batch
        bool prefault_pte = has_madv_populate &&
            (offsets[offsets.size()-1] / (float)offsets.size() * stride * sizeof(scalar_t) <= 4096);
        if (prefault_pte) {
            // try prefaulting the first range
            if (batch_populate_pte((void *)(src_p+starts_p[0]*stride), lengths_p[0]*stride*sizeof(scalar_t)) < 0) {
                // TORCH_WARN("Pre-faults on input tensor ranges unsupported.");
                prefault_pte = false;
            }
        }

        dynamic_parallel_for(0, starts.size(0), [&](int i) {
            long start = starts_p[i];
            long end = start + lengths_p[i];
            if (prefault_pte)
                batch_populate_pte((void*)(src_p + start*stride), lengths_p[i]*stride);
            std::copy(src_p + start*stride, src_p + end*stride, out_p + offsets[i]*stride);
        }, 1);
    });
    return out;
}

Tensor &ranges_gather(
    Tensor &out, const std::string &filename, const Tensor &starts, const Tensor &lengths
) {
    TORCH_CHECK(starts.scalar_type() == at::kLong, "index should have dtype long/int64");
    TORCH_CHECK(lengths.scalar_type() == at::kLong, "sizes should have dtype long/int64");
    // TORCH_CHECK(out.scalar_type() == src.scalar_type());
    // TORCH_CHECK(out.stride(0) == src.stride(0));

    int fd = open(filename.data(), O_RDONLY);
    TORCH_CHECK(fd > 0, "Fail to open ", filename)

    auto stride = out.stride(0);
    const auto *starts_p = starts.data_ptr<long>();
    const auto *lengths_p = lengths.data_ptr<long>();
    std::vector<long> offsets(lengths.size(0), 0);
    for (size_t i = 1; i < lengths.size(0); ++i) {
        offsets[i] += offsets[i-1] + lengths_p[i-1];
    }
    TORCH_CHECK(offsets[lengths.size(0)-1] <= out.size(0));

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::Bool, out.scalar_type(), "ranges_gather_dim0", [&]() {
        auto *out_p = out.data_ptr<scalar_t>();
        // std::cout << "(" << out.size(0) << ", " << stride << ")\n";
        // std::cout << "sizeof(scalar_t):" << sizeof(scalar_t) << "\n";

        dynamic_parallel_for(0, starts.size(0), [&](int i) {
            long nstart = starts_p[i] * stride;
            ssize_t nscalar = lengths_p[i] * stride;
            // std::cout << "[" << starts_p[i] << "+:" << lengths_p[i] << ")\n";
            ssize_t nbytes = pread(
                fd, out_p + offsets[i] * stride,
                nscalar * sizeof(scalar_t),
                nstart * sizeof(scalar_t)
            );
            TORCH_CHECK(nbytes == nscalar * sizeof(scalar_t), "Read less than expected: ", nbytes);
        }, 1);
    });

    TORCH_CHECK(close(fd) >= 0, "Fail to close ", filename);
    return out;
}

Tensor &ranges_add(
    Tensor &target, const Tensor &starts, const Tensor &lengths, const Tensor &values
) {
    TORCH_CHECK(starts.scalar_type() == at::kLong, "index should have dtype long/int64");
    TORCH_CHECK(lengths.scalar_type() == at::kLong, "sizes should have dtype long/int64");

    int num_par = omp_thread_count();
    const auto *starts_p = starts.data_ptr<long>();
    const auto *lengths_p = lengths.data_ptr<long>();
    dynamic_parallel_for(0, starts.size(0),
        [&](int i) {
            Tensor target_slice = target.slice(0, starts_p[i], starts_p[i]+lengths_p[i]);
            Tensor value_slice = values.index({i});
            target_slice.add_(value_slice);
        }, std::max((starts.size(0) / (2 * num_par)), 1L)
    );
    return target;
}

std::tuple<Tensor, Tensor> coo_list_merge(
    long num_nodes, const std::vector<EdgeType> &undirected
) {
    Tensor deg = torch::zeros({num_nodes}, torch::dtype(torch::kInt32));
    int *deg_ptr = deg.data_ptr<int>();

    // get degree counts
    dynamic_parallel_for(0, undirected.size(),
        [&](int i) {
            const auto &nodes = std::get<1>(undirected[i]);
            const auto *ptr = nodes.data_ptr<long>();
            long length = nodes.size(0);
            long ei = 0;
            long prev_nid = 0;
            int nid_count = 0;
            while (ei < length) {
                long nid = ptr[ei];
                if (nid == prev_nid) {
                    nid_count++;
                } else {
                    // new nid encountered
                    atomic_add(deg_ptr[prev_nid], nid_count);
                    prev_nid = nid;
                    nid_count = 1;
                }
                ++ei;
            }
            atomic_add(deg_ptr[prev_nid], nid_count);
        }, 1 // block_size = 1
    );

    // get rowptr
    Tensor rowptr = torch::zeros({num_nodes+1}, torch::dtype(torch::kInt64));
    Tensor rowout = rowptr.narrow(0, 1, num_nodes);
    // parallel_prefix_sum(num_nodes, deg_ptr, rowptr.data_ptr<long>());
    torch::cumsum_out(rowout, deg, 0);
    Tensor rowoff = rowptr.clone();
    auto *off_ptr = rowoff.data_ptr<long>();

    std::vector<size_t> sizes;
    std::transform(
        undirected.begin(), undirected.end(), std::back_inserter(sizes),
        [](const auto &ee) { return std::get<0>(ee).size(0); }
    );
    long num_edges = std::accumulate(sizes.begin(), sizes.end(), 0);
    Tensor src = torch::empty({num_edges}, torch::dtype(torch::kInt64));
    auto *src_out = src.data_ptr<long>();
    // Tensor dst = torch::empty({num_edges}, torch::dtype(torch::kInt64));
    // auto *dst_out = dst.data_ptr<long>();

    // scatter edges to their dst bins
    int num_par = omp_thread_count();
    dynamic_parallel_for(0, undirected.size(),
        [&](int i) {
            const auto &src_part = std::get<0>(undirected[i]);
            const auto &dst_part = std::get<1>(undirected[i]);
            const auto *src_ptr = src_part.data_ptr<long>();
            const auto *dst_ptr = dst_part.data_ptr<long>();
            long length = src_part.size(0);
            long ei = 0;
            long prev_nid = 0;
            long prev_off = 0;
            long nid_count = 0;
            while (ei < length) {
                long nid = dst_ptr[ei];
                if (nid == prev_nid) {
                    nid_count++;
                } else {
                    // new nid encountered
                    atomic_add(off_ptr[prev_nid], nid_count, prev_off);
                    std::copy(src_ptr + ei - nid_count, src_ptr + ei, src_out + prev_off);
                    // std::copy(dst_ptr + ei - nid_count, dst_ptr + ei, dst_out + prev_off);
                    prev_nid = nid;
                    nid_count = 1;
                }
                ++ei;
            }
            atomic_add(off_ptr[prev_nid], nid_count, prev_off);
            std::copy(src_ptr + ei - nid_count, src_ptr + ei, src_out + prev_off);
            // std::copy(dst_ptr + ei - nid_count, dst_ptr + ei, dst_out + prev_off);
        }, std::max((int)undirected.size()/(2*num_par), 1)  // block_size = 1
    );

    return {rowptr, src};
}


std::tuple<Tensor, Tensor> coo_ranges_merge(
    long num_nodes,
    const std::vector<EdgeType> &coo_list,
    const std::vector<Tensor> &starts,
    const std::vector<Tensor> &sizes
) {
    long num_tensors = 0;
    for (const auto t : starts) num_tensors += t.size(0);

    using index_t = long;
    std::vector<EdgeType> consolidated(num_tensors); // consolidated.reserve(num_tensors);
    size_t tensor_idx = 0;
    for (size_t i = 0; i < coo_list.size(); ++i) {
        const auto &src = std::get<0>(coo_list[i]);
        const auto &dst = std::get<1>(coo_list[i]);
        const auto *src_p = src.data_ptr<index_t>();
        const auto *dst_p = dst.data_ptr<index_t>();
        const auto &coo_starts = starts[i];
        const auto &coo_sizes = sizes[i];
        const auto *starts_p = coo_starts.data_ptr<long>();
        const auto *sizes_p = coo_sizes.data_ptr<long>();
        // for (size_t j = 0; j < coo_starts.size(0); ++j) {
        //     consolidated.push_back({
        //         src.slice(0, starts_p[j], starts_p[j]+sizes_p[j]),
        //         dst.slice(0, starts_p[j], starts_p[j]+sizes_p[j])
        //     });
        // }
        dynamic_parallel_for(0, coo_starts.size(0), [&](int j) {
            EdgeType frag {
                src.slice(0, starts_p[j], starts_p[j]+sizes_p[j]),
                dst.slice(0, starts_p[j], starts_p[j]+sizes_p[j])
            };
            std::swap(consolidated[tensor_idx + j], frag);
            if (has_madv_populate && sizes_p[j] > 4096) {
                batch_populate_pte((void*)(src_p + starts_p[j]), sizes_p[j] * sizeof(index_t));
                batch_populate_pte((void*)(dst_p + starts_p[j]), sizes_p[j] * sizeof(index_t));
            }
        });
        tensor_idx += coo_starts.size(0);
    }

    return coo_list_merge(num_nodes, consolidated);
}

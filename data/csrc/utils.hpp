#pragma once
#include <iostream>
#include <vector>
#include <torch/torch.h>

template <typename scalar_t>
inline torch::Tensor vector_to_tensor(const std::vector<scalar_t>& vec,
                                      bool pin_memory = false) {
  auto tensor = torch::empty(
      vec.size(), torch::TensorOptions()
                      .dtype(torch::CppTypeToScalarType<scalar_t>::value)
                      .device(torch::kCPU)
                      .layout(torch::kStrided)
                      .pinned_memory(pin_memory)
                      .requires_grad(false));
  const auto tensor_data = tensor.template data_ptr<scalar_t>();
  std::copy(vec.begin(), vec.end(), tensor_data);
  return tensor;
}

template <typename T>
struct Slice {
    const T *begin_, *end_;
    size_t size_;

    Slice(const T *begin, const T *end)
    : begin_(begin), end_(end), size_(end_-begin_)
    {}
    Slice(const T *begin, size_t size)
    : begin_(begin), end_(begin+size), size_(size)
    {}
    size_t size() const { return size_; }
    const T *begin() const { return begin_; }
    const T *end() const { return end_; }
    const T operator[](size_t idx) const {
        return *(this->begin() + idx);
    }
    std::vector<T> to_vec() const {
        std::vector<T> vec(size_);
        std::copy(begin_, end_, vec.begin());
        return vec;
    }

    static Slice from_tensor(torch::Tensor tensor) {
        auto raw_data = tensor.data_ptr<T>();
        size_t len = tensor.numel();
        return Slice<T>(raw_data, len);
    }
};

template <typename scalar_t>
inline std::vector<scalar_t> tensor_to_vector(torch::Tensor tensor) {
    return Slice<scalar_t>::from_tensor(tensor).to_vec();
}

namespace std {
template <typename T>
ostream &operator<<(ostream &os, const vector<T> &vec) {
    os << '[';
    for (const auto v : vec) {
        os << ' ' << v; 
    }
    os << ']';
    return os;
}
}

/**
 * Parallel utilities
*/
#include <omp.h>

static int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

template <class F>
void dynamic_parallel_for(long start, long end, const F &f, int block_size=64, int num_par=0) {
  #ifdef NDEBUG
    if (num_par == 0) {
        #pragma omp parallel for schedule(dynamic, block_size)
        for (long i = start; i < end; ++i) {
            f(i);
        }
    } else {
        #pragma omp parallel num_threads(num_par)
        {
            #pragma omp for schedule(dynamic, block_size)
            for (long i = start; i < end; ++i) {
                f(i);
            }
        }
    }
  #else
    for (long i = start; i < end; ++i) {
        f(i);
    }
  #endif
}

template <typename T>
inline void atomic_add(T &obj, const T &val) {
    #pragma omp atomic update
    obj += val;
}

template <typename T>
inline void atomic_add(T &obj, const T &val, T &out) {
    #pragma omp atomic capture
    {
        out = obj;
        obj += val;
    }
}

template <typename InTy = unsigned, typename OutTy = unsigned>
OutTy parallel_prefix_sum(size_t length, const InTy *in, OutTy *out) {
    const size_t block_size = 1 << 16;
    const size_t num_blocks = (length + block_size - 1) / block_size;
    std::vector<OutTy> local_sums(num_blocks);
    // count how many bits are set on each thread
    #pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block ++) {
        OutTy lsum = 0;
        size_t block_end = std::min((block + 1) * block_size, length);
        for (size_t i = block * block_size; i < block_end; i++)
            lsum += in[i];
        local_sums[block] = lsum;
    }
    std::vector<OutTy> bulk_prefix(num_blocks + 1);
    OutTy total = 0;
    for (size_t block = 0; block < num_blocks; block++) {
        bulk_prefix[block] = total;
        total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    #pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block ++) {
        OutTy local_total = bulk_prefix[block];
        size_t block_end = std::min((block + 1) * block_size, length);
        for (size_t i = block * block_size; i < block_end; i++) {
            out[i] = local_total;
            local_total += in[i];
        }
    }
    out[length] = bulk_prefix[num_blocks];
    return out[length];
}

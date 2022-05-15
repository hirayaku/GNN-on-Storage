#pragma once

#include <tuple>
#include <iostream>
#include <torch/torch.h>

#define CHECK_CPU(x) \
  TORCH_INTERNAL_ASSERT(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) TORCH_INTERNAL_ASSERT(x, "Input mismatch")

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

template <typename scalar_t>
class tensor_iter {
public:
    tensor_iter(torch::Tensor &t)
    : ptr(t.data_ptr<scalar_t>()), size(t.numel())
    {}
    scalar_t *begin() const { return ptr; }
    scalar_t *end() const { return ptr + size; }
public:
    scalar_t *ptr;
    long size;
};

template<typename Type, unsigned N, unsigned Last>
struct tuple_printer {
    static void print(std::ostream& out, const Type& value) {
        out << std::get<N>(value) << ", ";
        tuple_printer<Type, N + 1, Last>::print(out, value);
    }
};

template<typename Type, unsigned N>
struct tuple_printer<Type, N, N> {

    static void print(std::ostream& out, const Type& value) {
        out << std::get<N>(value);
    }

};

template<typename... Types>
std::ostream& operator<<(std::ostream& out, const std::tuple<Types...>& value) {
    out << "(";
    tuple_printer<std::tuple<Types...>, 0, sizeof...(Types) - 1>::print(out, value);
    out << ")";
    return out;
}

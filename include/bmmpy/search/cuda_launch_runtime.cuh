#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace bmmpy::cuda_launch_detail {

inline void check_cuda(const cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + context + ": " +
                                 cudaGetErrorString(status));
    }
}

inline std::size_t next_capacity(const std::size_t current, const std::size_t required) {
    if (current >= required)
        return current;

    std::size_t capacity = current == 0 ? 256 : current;
    while (capacity < required) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            return required;
        capacity *= 2;
    }
    return capacity;
}

template <typename T>
inline void
ensure_buffer(T*& ptr, std::size_t& capacity, const std::size_t required, const char* label) {
    if (required <= capacity)
        return;

    if (ptr != nullptr)
        check_cuda(cudaFree(ptr), "cudaFree(realloc)");

    const std::size_t new_capacity = next_capacity(capacity, required);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), new_capacity * sizeof(T)), label);
    capacity = new_capacity;
}

} // namespace bmmpy::cuda_launch_detail
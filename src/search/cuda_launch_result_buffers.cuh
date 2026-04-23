#pragma once

#include "cuda_launch_runtime.cuh"
#include "cuda_launch_topk.cuh"

#include <cstddef>
#include <cuda_runtime.h>

namespace bmmpy::cuda_launch_detail {

struct DeviceResultBuffers {
    DeviceTopKEntry* d_block_results = nullptr;
    DeviceTopKEntry* d_out_results = nullptr;

    std::size_t block_results_capacity = 0;
    std::size_t out_results_capacity = 0;

    void reset() noexcept {
        if (d_block_results != nullptr)
            (void)cudaFree(d_block_results);
        if (d_out_results != nullptr)
            (void)cudaFree(d_out_results);

        d_block_results = nullptr;
        d_out_results = nullptr;
        block_results_capacity = 0;
        out_results_capacity = 0;
    }
};

inline void ensure_result_buffers(DeviceResultBuffers& buffers,
                                  const std::size_t block_result_count,
                                  const std::size_t out_result_count) {
    ensure_buffer(buffers.d_block_results,
                  buffers.block_results_capacity,
                  block_result_count,
                  "cudaMalloc(block_results)");
    ensure_buffer(buffers.d_out_results,
                  buffers.out_results_capacity,
                  out_result_count,
                  "cudaMalloc(out_results)");
}

} // namespace bmmpy::cuda_launch_detail
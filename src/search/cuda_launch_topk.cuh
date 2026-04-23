// src/search/cuda_launch_topk.cuh
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace bmmpy::cuda_launch_detail {

inline constexpr int kMaxCandidates = 128;

struct DeviceTopKEntry {
    std::uint64_t mask;
    std::uint32_t weight;
};

__host__ __device__ inline bool candidate_less(const DeviceTopKEntry& lhs,
                                               const DeviceTopKEntry& rhs) noexcept {
    return lhs.weight < rhs.weight || (lhs.weight == rhs.weight && lhs.mask < rhs.mask);
}

template <int K>
__device__ inline void
insert_topk(DeviceTopKEntry (&best)[K], int& size, const DeviceTopKEntry incoming) {
    if (incoming.mask == 0)
        return;

    if (size < K) {
        best[size++] = incoming;
    } else if (!candidate_less(incoming, best[size - 1])) {
        return;
    } else {
        best[size - 1] = incoming;
    }

    for (int i = size - 1; i > 0; --i) {
        if (!candidate_less(best[i], best[i - 1]))
            break;

        const DeviceTopKEntry tmp = best[i - 1];
        best[i - 1] = best[i];
        best[i] = tmp;
    }
}

static __global__ void merge_topk_kernel(const DeviceTopKEntry* block_results,
                                         const std::size_t block_count,
                                         const std::size_t k,
                                         DeviceTopKEntry* out_results) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KStatic = kMaxCandidates;
    DeviceTopKEntry best[KStatic];
    int size = 0;
    const std::size_t kk =
        k < static_cast<std::size_t>(KStatic) ? k : static_cast<std::size_t>(KStatic);

    for (std::size_t block = 0; block < block_count; ++block) {
        const std::size_t base = block * kk;
        for (std::size_t i = 0; i < kk; ++i)
            insert_topk<KStatic>(best, size, block_results[base + i]);
    }

    for (std::size_t i = 0; i < kk; ++i) {
        if (i < static_cast<std::size_t>(size))
            out_results[i] = best[i];
        else
            out_results[i] = DeviceTopKEntry{0, 0};
    }
}

} // namespace bmmpy::cuda_launch_detail
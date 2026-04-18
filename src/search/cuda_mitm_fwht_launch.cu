#include "bmmpy/search/cuda_mitm_fwht_launch.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace bmmpy {
namespace {
struct DeviceTopKEntry {
    std::uint64_t mask;
    std::uint32_t weight;
};

__device__ inline bool candidate_less(const DeviceTopKEntry& lhs, const DeviceTopKEntry& rhs) {
    return lhs.weight < rhs.weight || (lhs.weight == rhs.weight && lhs.mask < rhs.mask);
}

template <int32_t K>
__device__ inline void
insert_topk(DeviceTopKEntry (&best)[K], int32_t& size, DeviceTopKEntry incoming) {
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

__device__ inline int parity64(std::uint64_t value) {
    value ^= value >> 32;
    value ^= value >> 16;
    value ^= value >> 8;
    value ^= value >> 4;
    value &= 0xFULL;
    return (0x6996u >> value) & 1u;
}

__global__ void sweep_kernel(const std::uint64_t* q,
                             const std::uint64_t* r,
                             const std::int32_t* multiplicity,
                             std::size_t unique_count,
                             std::size_t low_bits,
                             std::size_t high_bits,
                             std::int32_t total_weight,
                             std::size_t k,
                             DeviceTopKEntry* block_results) {
    const std::uint64_t x = static_cast<std::uint64_t>(blockIdx.x);
    const std::size_t n_low = std::size_t{1} << low_bits;

    extern __shared__ std::int16_t g[];

    for (std::size_t i = threadIdx.x; i < n_low; i += blockDim.x)
        g[i] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (std::size_t i = 0; i < unique_count; ++i) {
            const int sign = parity64(x & q[i]) ? -1 : 1;
            g[r[i]] = static_cast<std::int16_t>(g[r[i]] + sign * multiplicity[i]);
        }
    }
    __syncthreads();

    for (std::size_t len = 1; len < n_low; len <<= 1) {
        const std::size_t step = len << 1;
        const std::size_t butterflies = n_low >> 1;

        for (std::size_t idx = threadIdx.x; idx < butterflies; idx += blockDim.x) {
            const std::size_t group = idx / len;
            const std::size_t offset = idx % len;
            const std::size_t left = group * step + offset;
            const std::size_t right = left + len;

            const std::int16_t a = g[left];
            const std::int16_t b = g[right];
            g[left] = static_cast<std::int16_t>(a + b);
            g[right] = static_cast<std::int16_t>(a - b);
        }

        __syncthreads();
    }

    if (threadIdx.x == 0) {
        constexpr int KStatic = 64;
        DeviceTopKEntry best[KStatic];
        int size = 0;
        const std::size_t kk = min(k, static_cast<std::size_t>(KStatic));

        for (std::size_t y = 0; y < n_low; ++y) {
            const std::int32_t h = g[y];
            const std::int64_t w2 =
                static_cast<std::int64_t>(total_weight) - static_cast<std::int64_t>(h);

            if (w2 < 0 || (w2 & 1) != 0)
                continue;

            const std::uint64_t mask = (x << low_bits) | static_cast<std::uint64_t>(y);
            if (mask == 0)
                continue;

            insert_topk<KStatic>(
                best, size, DeviceTopKEntry{mask, static_cast<std::uint32_t>(w2 / 2)});
        }

        const std::size_t out_base = static_cast<std::size_t>(blockIdx.x) * kk;
        for (std::size_t i = 0; i < kk; ++i) {
            if (i < static_cast<std::size_t>(size))
                block_results[out_base + i] = best[i];
            else
                block_results[out_base + i] = DeviceTopKEntry{0, 0};
        }
    }
}

__global__ void merge_kernel(const DeviceTopKEntry* block_results,
                             std::size_t block_count,
                             std::size_t k,
                             DeviceTopKEntry* out_results) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KStatic = 64;
    DeviceTopKEntry best[KStatic];
    int size = 0;
    const std::size_t kk = min(k, static_cast<std::size_t>(KStatic));

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

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + context + ": " +
                                 cudaGetErrorString(status));
    }
}

} // namespace

std::vector<CudaMitmFwhtResult> run_cuda_mitm_fwht_search(const CompactSplitWindow& prep,
                                                          std::size_t max_candidates) {
    if (max_candidates == 0 || prep.multiplicity.empty())
        return {};

    if (max_candidates > 64) {
        throw std::invalid_argument("run_cuda_mitm_fwht_search: max_candidates must be <= 64");
    }

    const std::size_t block_count = std::size_t{1} << prep.high_bits;
    const std::size_t k = max_candidates;
    const std::size_t n_low = std::size_t{1} << prep.low_bits;
    const std::size_t shared_bytes = n_low * sizeof(std::int16_t);

    std::uint64_t* d_q = nullptr;
    std::uint64_t* d_r = nullptr;
    std::int32_t* d_m = nullptr;
    DeviceTopKEntry* d_block_results = nullptr;
    DeviceTopKEntry* d_out_results = nullptr;

    check_cuda(cudaMalloc(&d_q, prep.q.size() * sizeof(std::uint64_t)), "cudaMalloc(q)");
    check_cuda(cudaMalloc(&d_r, prep.r.size() * sizeof(std::uint64_t)), "cudaMalloc(r)");
    check_cuda(cudaMalloc(&d_m, prep.multiplicity.size() * sizeof(std::int32_t)),
               "cudaMalloc(multiplicity)");
    check_cuda(cudaMalloc(&d_block_results, block_count * k * sizeof(DeviceTopKEntry)),
               "cudaMalloc(block_results)");
    check_cuda(cudaMalloc(&d_out_results, k * sizeof(DeviceTopKEntry)), "cudaMalloc(out_results)");

    check_cuda(
        cudaMemcpy(
            d_q, prep.q.data(), prep.q.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
        "cudaMemcpy(q)");
    check_cuda(
        cudaMemcpy(
            d_r, prep.r.data(), prep.r.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
        "cudaMemcpy(r)");
    check_cuda(cudaMemcpy(d_m,
                          prep.multiplicity.data(),
                          prep.multiplicity.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(multiplicity)");

    constexpr int threads = 256;
    sweep_kernel<<<static_cast<unsigned int>(block_count), threads, shared_bytes>>>(
        d_q,
        d_r,
        d_m,
        prep.multiplicity.size(),
        prep.low_bits,
        prep.high_bits,
        prep.total_weight,
        k,
        d_block_results);
    check_cuda(cudaGetLastError(), "sweep_kernel launch");

    merge_kernel<<<1, 1>>>(d_block_results, block_count, k, d_out_results);
    check_cuda(cudaGetLastError(), "merge_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<DeviceTopKEntry> host_out(k);
    check_cuda(
        cudaMemcpy(
            host_out.data(), d_out_results, k * sizeof(DeviceTopKEntry), cudaMemcpyDeviceToHost),
        "cudaMemcpy(out_results)");

    cudaFree(d_q);
    cudaFree(d_r);
    cudaFree(d_m);
    cudaFree(d_block_results);
    cudaFree(d_out_results);

    std::vector<CudaMitmFwhtResult> out;
    out.reserve(k);
    for (const DeviceTopKEntry& entry : host_out) {
        if (entry.mask != 0)
            out.push_back(CudaMitmFwhtResult{entry.mask, entry.weight});
    }
    return out;
}

} // namespace bmmpy
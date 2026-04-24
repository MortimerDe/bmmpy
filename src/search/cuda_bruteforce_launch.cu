#include "bmmpy/search/cuda_bruteforce_launch.hpp"
#include "bmmpy/search/cuda_launch_result_buffers.cuh"
#include "bmmpy/search/cuda_launch_runtime.cuh"
#include "bmmpy/search/cuda_launch_topk.cuh"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

using cuda_launch_detail::candidate_less;
using cuda_launch_detail::check_cuda;
using cuda_launch_detail::DeviceResultBuffers;
using cuda_launch_detail::DeviceTopKEntry;
using cuda_launch_detail::ensure_buffer;
using cuda_launch_detail::ensure_result_buffers;
using cuda_launch_detail::kMaxCandidates;

constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kAutoChunkBits = 12;
constexpr std::size_t kSupportedWordsPerRow512 = 8;
constexpr std::size_t kSupportedWordsPerRow1024 = 16;
constexpr std::size_t kMaxRows = 64;
constexpr std::size_t kMaskBits = 64;

struct BruteforceDeviceWorkspace {
    int device = -1;

    std::uint64_t* d_row_words = nullptr;
    std::size_t row_words_capacity = 0;

    DeviceResultBuffers results;

    ~BruteforceDeviceWorkspace() { reset(); }

    void reset() noexcept {
        if (d_row_words != nullptr)
            (void)cudaFree(d_row_words);

        d_row_words = nullptr;
        row_words_capacity = 0;

        results.reset();
        device = -1;
    }
};

bool is_supported_words_per_row(const std::size_t words_per_row) noexcept {
    return words_per_row == kSupportedWordsPerRow512 || words_per_row == kSupportedWordsPerRow1024;
}

std::size_t resolve_chunk_bits(const std::size_t rows, const std::size_t configured_chunk_bits) {
    if (rows == 0)
        return 0;

    if (configured_chunk_bits == 0) {
        if (rows == 1)
            return 1;

        return std::min<std::size_t>(rows - 1, kAutoChunkBits);
    }

    if (configured_chunk_bits >= rows || configured_chunk_bits >= kMaskBits) {
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: chunk_bits must be in [1, rows - 1] or 0 for auto");
    }

    return configured_chunk_bits;
}

std::uint64_t prefix_count_for(const std::size_t rows, const std::size_t chunk_bits) {
    const std::size_t high_bits = rows - chunk_bits;
    if (high_bits >= kMaskBits) {
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: high-prefix bit count must be < 64");
    }

    return high_bits == 0 ? std::uint64_t{1} : (std::uint64_t{1} << high_bits);
}

__device__ inline unsigned int ctz64_device(const std::uint64_t value) {
    return static_cast<unsigned int>(__ffsll(static_cast<unsigned long long>(value)) - 1);
}

__device__ inline void
insert_topk_runtime(DeviceTopKEntry* best, int& size, const int k, const DeviceTopKEntry incoming) {
    if (incoming.mask == 0 || k <= 0)
        return;

    if (size < k) {
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

template <int WordCount> __device__ inline void zero_words(std::uint64_t (&words)[WordCount]) {
#pragma unroll
    for (int i = 0; i < WordCount; ++i)
        words[i] = 0;
}

template <int WordCount>
__device__ inline void xor_row_into(std::uint64_t (&current)[WordCount],
                                    const std::uint64_t* __restrict__ row_words,
                                    const std::uint32_t row) {
    const std::uint64_t* src =
        row_words + static_cast<std::size_t>(row) * static_cast<std::size_t>(WordCount);

#pragma unroll
    for (int i = 0; i < WordCount; ++i)
        current[i] ^= src[i];
}

template <int WordCount>
__device__ inline std::uint32_t popcount_words(const std::uint64_t (&words)[WordCount]) {
    std::uint32_t total = 0;

#pragma unroll
    for (int i = 0; i < WordCount; ++i) {
        total += static_cast<std::uint32_t>(__popcll(static_cast<unsigned long long>(words[i])));
    }

    return total;
}

template <int WordCount>
__global__ void bruteforce_persistent_kernel(const std::uint64_t* __restrict__ row_words,
                                             const std::uint32_t rows,
                                             const std::uint32_t chunk_bits,
                                             const std::uint64_t prefix_count,
                                             const std::uint32_t k,
                                             DeviceTopKEntry* __restrict__ block_results) {
    constexpr int KStatic = kMaxCandidates;
    const int kk = min(static_cast<int>(k), KStatic);

    DeviceTopKEntry local_best[KStatic];
    int local_size = 0;

    std::uint64_t current[WordCount];

    const std::uint32_t high_bits = rows - chunk_bits;
    const std::uint64_t low_states = std::uint64_t{1} << chunk_bits;

    const std::uint64_t first_prefix =
        static_cast<std::uint64_t>(blockIdx.x) * static_cast<std::uint64_t>(blockDim.x) +
        static_cast<std::uint64_t>(threadIdx.x);
    const std::uint64_t prefix_stride =
        static_cast<std::uint64_t>(gridDim.x) * static_cast<std::uint64_t>(blockDim.x);

    for (std::uint64_t prefix = first_prefix; prefix < prefix_count; prefix += prefix_stride) {
        zero_words<WordCount>(current);

        for (std::uint32_t high = 0; high < high_bits; ++high) {
            if (((prefix >> high) & std::uint64_t{1}) != 0)
                xor_row_into<WordCount>(current, row_words, chunk_bits + high);
        }

        std::uint64_t current_mask = prefix << chunk_bits;
        if (current_mask != 0) {
            insert_topk_runtime(local_best,
                                local_size,
                                kk,
                                DeviceTopKEntry{current_mask, popcount_words<WordCount>(current)});
        }

        for (std::uint64_t step = 1; step < low_states; ++step) {
            const std::uint32_t bit = static_cast<std::uint32_t>(ctz64_device(step));

            xor_row_into<WordCount>(current, row_words, bit);
            current_mask ^= (std::uint64_t{1} << bit);

            insert_topk_runtime(local_best,
                                local_size,
                                kk,
                                DeviceTopKEntry{current_mask, popcount_words<WordCount>(current)});
        }
    }

    __shared__ DeviceTopKEntry shared_best[KStatic];
    __shared__ DeviceTopKEntry thread_stage[kThreadsPerBlock];
    __shared__ int shared_best_size;

    if (threadIdx.x == 0)
        shared_best_size = 0;
    __syncthreads();

    for (int round = 0; round < kk; ++round) {
        DeviceTopKEntry staged{0, 0};
        if (round < local_size)
            staged = local_best[round];

        thread_stage[threadIdx.x] = staged;
        __syncthreads();

        if (threadIdx.x == 0) {
            int merged_size = shared_best_size;

            for (int tid = 0; tid < blockDim.x; ++tid)
                insert_topk_runtime(shared_best, merged_size, kk, thread_stage[tid]);

            shared_best_size = merged_size;
        }

        __syncthreads();
    }

    const std::size_t out_base =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(kk);

    for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x);
         i < static_cast<std::uint32_t>(kk);
         i += static_cast<std::uint32_t>(blockDim.x)) {
        if (i < static_cast<std::uint32_t>(shared_best_size))
            block_results[out_base + i] = shared_best[i];
        else
            block_results[out_base + i] = DeviceTopKEntry{0, 0};
    }
}

template <int WordCount> std::size_t recommend_persistent_blocks(const std::uint64_t prefix_count) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int active_blocks_per_sm = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm, bruteforce_persistent_kernel<WordCount>, kThreadsPerBlock, 0),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    const std::uint64_t required_blocks =
        (prefix_count + static_cast<std::uint64_t>(kThreadsPerBlock) - 1) /
        static_cast<std::uint64_t>(kThreadsPerBlock);

    const std::uint64_t desired_blocks =
        static_cast<std::uint64_t>(std::max(prop.multiProcessorCount, 1)) *
        static_cast<std::uint64_t>(std::max(active_blocks_per_sm, 1));

    const std::uint64_t block_count =
        std::min(required_blocks, std::max<std::uint64_t>(desired_blocks, 1));

    return static_cast<std::size_t>(std::max<std::uint64_t>(block_count, 1));
}

template <int WordCount>
void launch_bruteforce_kernel(BruteforceDeviceWorkspace& workspace,
                              const std::size_t rows,
                              const std::size_t chunk_bits,
                              const std::uint64_t prefix_count,
                              const std::size_t k) {
    const std::size_t persistent_blocks = recommend_persistent_blocks<WordCount>(prefix_count);

    ensure_result_buffers(workspace.results, persistent_blocks * k, k);

    bruteforce_persistent_kernel<WordCount>
        <<<static_cast<unsigned int>(persistent_blocks), kThreadsPerBlock>>>(
            workspace.d_row_words,
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(chunk_bits),
            prefix_count,
            static_cast<std::uint32_t>(k),
            workspace.results.d_block_results);
    check_cuda(cudaGetLastError(), "bruteforce_persistent_kernel launch");

    cuda_launch_detail::merge_topk_kernel<<<1, 1>>>(
        workspace.results.d_block_results, persistent_blocks, k, workspace.results.d_out_results);
    check_cuda(cudaGetLastError(), "merge_topk_kernel launch");
}

void dispatch_bruteforce_kernel(const CudaBruteforcePlan& plan,
                                BruteforceDeviceWorkspace& workspace,
                                const std::size_t chunk_bits,
                                const std::uint64_t prefix_count,
                                const std::size_t k) {
    switch (plan.words_per_row) {
    case kSupportedWordsPerRow512:
        launch_bruteforce_kernel<8>(workspace, plan.rows, chunk_bits, prefix_count, k);
        break;
    case kSupportedWordsPerRow1024:
        launch_bruteforce_kernel<16>(workspace, plan.rows, chunk_bits, prefix_count, k);
        break;
    default:
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: only widths 512 and 1024 are supported");
    }
}

BruteforceDeviceWorkspace& get_workspace() {
    thread_local BruteforceDeviceWorkspace workspace;

    int current_device = 0;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");

    if (workspace.device != current_device) {
        workspace.reset();
        workspace.device = current_device;
    }

    return workspace;
}

void upload_plan(const CudaBruteforcePlan& plan, BruteforceDeviceWorkspace& workspace) {
    ensure_buffer(workspace.d_row_words,
                  workspace.row_words_capacity,
                  plan.row_words.size(),
                  "cudaMalloc(cuda_bruteforce_row_words)");

    check_cuda(cudaMemcpy(workspace.d_row_words,
                          plan.row_words.data(),
                          plan.row_words.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(cuda_bruteforce_row_words)");
}

} // namespace

std::vector<CudaBruteforceResult> run_cuda_bruteforce_search(const CudaBruteforcePlan& plan,
                                                             const std::size_t max_candidates) {
    if (max_candidates == 0 || plan.rows == 0 || plan.words_per_row == 0)
        return {};

    if (max_candidates > static_cast<std::size_t>(kMaxCandidates)) {
        throw std::invalid_argument("run_cuda_bruteforce_search: max_candidates must be <= 128");
    }

    if (plan.rows > kMaxRows) {
        throw std::invalid_argument("run_cuda_bruteforce_search: row count must be <= 64");
    }

    if (!is_supported_words_per_row(plan.words_per_row)) {
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: only widths 512 and 1024 are supported");
    }

    if (plan.row_words.size() != plan.rows * plan.words_per_row) {
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: row_words size does not match rows * words_per_row");
    }

    if (plan.chunk_bits != 0 && plan.chunk_bits >= plan.rows) {
        throw std::invalid_argument(
            "run_cuda_bruteforce_search: chunk_bits must be in [1, rows - 1] or 0 for auto");
    }

    const std::size_t chunk_bits = resolve_chunk_bits(plan.rows, plan.chunk_bits);
    const std::uint64_t prefix_count = prefix_count_for(plan.rows, chunk_bits);

    BruteforceDeviceWorkspace& workspace = get_workspace();
    upload_plan(plan, workspace);

    dispatch_bruteforce_kernel(plan, workspace, chunk_bits, prefix_count, max_candidates);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return {};
}

} // namespace bmmpy
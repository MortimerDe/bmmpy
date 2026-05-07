#include "bmmpy/search/cuda_launch_result_buffers.cuh"
#include "bmmpy/search/cuda_launch_runtime.cuh"
#include "bmmpy/search/cuda_launch_topk.cuh"
#include "bmmpy/search/cuda_mitm_fwht_launch.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

using cuda_launch_detail::check_cuda;
using cuda_launch_detail::DeviceResultBuffers;
using cuda_launch_detail::DeviceTopKEntry;
using cuda_launch_detail::ensure_buffer;
using cuda_launch_detail::ensure_result_buffers;
using cuda_launch_detail::insert_topk;
using cuda_launch_detail::kMaxCandidates;

constexpr int k_threads_per_block = 256;
constexpr int k_warp_size = 32;
constexpr int k_warp_count = k_threads_per_block / k_warp_size;
constexpr std::size_t k_max_specialized_low_bits = 15;

static_assert(k_threads_per_block % k_warp_size == 0,
              "k_threads_per_block must be a multiple of warp size");

struct HostGroupedPlan {
    std::size_t low_bits = 0;
    std::size_t high_bits = 0;
    std::int32_t total_weight = 0;

    std::vector<std::uint32_t> r_offsets;
    std::vector<std::uint64_t> q;
    std::vector<std::uint16_t> multiplicity;
};

struct MitmDeviceWorkspace {
    int device = -1;

    std::uint32_t* d_r_offsets = nullptr;
    std::uint64_t* d_q = nullptr;
    std::uint16_t* d_multiplicity = nullptr;

    std::size_t r_offsets_capacity = 0;
    std::size_t q_capacity = 0;
    std::size_t multiplicity_capacity = 0;

    DeviceResultBuffers results;

    ~MitmDeviceWorkspace() { reset(); }

    void reset() noexcept {
        if (d_r_offsets != nullptr)
            (void)cudaFree(d_r_offsets);
        if (d_q != nullptr)
            (void)cudaFree(d_q);
        if (d_multiplicity != nullptr)
            (void)cudaFree(d_multiplicity);

        d_r_offsets = nullptr;
        d_q = nullptr;
        d_multiplicity = nullptr;

        r_offsets_capacity = 0;
        q_capacity = 0;
        multiplicity_capacity = 0;

        results.reset();
        device = -1;
    }
};

__device__ inline int parity64(const std::uint64_t value) {
    return static_cast<int>(__popcll(static_cast<unsigned long long>(value)) & 1ull);
}

template <int LowBits> __device__ inline void fwht_shared_inplace(std::int16_t* g) {
    constexpr std::uint32_t n_low = std::uint32_t{1} << LowBits;

#pragma unroll
    for (std::uint32_t len = 1; len < n_low; len <<= 1) {
        const std::uint32_t step = len << 1;
        const std::uint32_t butterflies = n_low >> 1;

        for (std::uint32_t idx = static_cast<std::uint32_t>(threadIdx.x); idx < butterflies;
             idx += static_cast<std::uint32_t>(blockDim.x)) {
            const std::uint32_t group = idx / len;
            const std::uint32_t offset = idx % len;
            const std::uint32_t left = group * step + offset;
            const std::uint32_t right = left + len;

            const std::int16_t a = g[left];
            const std::int16_t b = g[right];
            g[left] = static_cast<std::int16_t>(a + b);
            g[right] = static_cast<std::int16_t>(a - b);
        }

        __syncthreads();
    }
}

template <int LowBits>
__global__ void sweep_persistent_kernel(const std::uint32_t* __restrict__ r_offsets,
                                        const std::uint64_t* __restrict__ q,
                                        const std::uint16_t* __restrict__ multiplicity,
                                        const std::uint64_t high_state_count,
                                        const std::int32_t total_weight,
                                        const std::uint32_t k,
                                        DeviceTopKEntry* __restrict__ block_results) {
    constexpr std::uint32_t n_low = std::uint32_t{1} << LowBits;
    constexpr int KStatic = kMaxCandidates;

    extern __shared__ std::int16_t g[];

    __shared__ DeviceTopKEntry shared_best[KStatic];
    __shared__ int shared_best_size;
    __shared__ DeviceTopKEntry warp_best[k_warp_count][KStatic];
    __shared__ int warp_best_size[k_warp_count];
    __shared__ DeviceTopKEntry warp_stage[k_warp_count][k_warp_size];

    const int warp_id = static_cast<int>(threadIdx.x) / k_warp_size;
    const int lane_id = static_cast<int>(threadIdx.x) % k_warp_size;

    if (threadIdx.x == 0)
        shared_best_size = 0;
    __syncthreads();

    for (std::uint64_t x = static_cast<std::uint64_t>(blockIdx.x); x < high_state_count;
         x += static_cast<std::uint64_t>(gridDim.x)) {
        for (std::uint32_t r = static_cast<std::uint32_t>(threadIdx.x); r < n_low;
             r += static_cast<std::uint32_t>(blockDim.x)) {
            std::int32_t acc = 0;
            const std::uint32_t start = r_offsets[r];
            const std::uint32_t end = r_offsets[r + 1];

            for (std::uint32_t idx = start; idx < end; ++idx) {
                const int sign = parity64(x & q[idx]) ? -1 : 1;
                acc += sign * static_cast<std::int32_t>(multiplicity[idx]);
            }

            g[r] = static_cast<std::int16_t>(acc);
        }
        __syncthreads();

        fwht_shared_inplace<LowBits>(g);

        if (lane_id == 0)
            warp_best_size[warp_id] = 0;
        __syncthreads();

        for (std::uint32_t y_base = static_cast<std::uint32_t>(warp_id * k_warp_size);
             y_base < n_low;
             y_base += static_cast<std::uint32_t>(k_warp_count * k_warp_size)) {
            const std::uint32_t y = y_base + static_cast<std::uint32_t>(lane_id);

            DeviceTopKEntry candidate{0, 0};
            if (y < n_low) {
                const std::int32_t h = static_cast<std::int32_t>(g[y]);
                const std::int64_t w2 =
                    static_cast<std::int64_t>(total_weight) - static_cast<std::int64_t>(h);

                if (w2 >= 0 && (w2 & 1) == 0) {
                    const std::uint64_t mask = (x << LowBits) | static_cast<std::uint64_t>(y);
                    if (mask != 0) {
                        candidate.mask = mask;
                        candidate.weight = static_cast<std::uint32_t>(w2 / 2);
                    }
                }
            }

            warp_stage[warp_id][lane_id] = candidate;
            __syncwarp();

            if (lane_id == 0) {
                for (int lane = 0; lane < k_warp_size; ++lane) {
                    insert_topk<KStatic>(
                        warp_best[warp_id], warp_best_size[warp_id], warp_stage[warp_id][lane]);
                }
            }

            __syncwarp();
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            DeviceTopKEntry best_local[KStatic];
            int local_size = shared_best_size;
            for (int i = 0; i < local_size; ++i)
                best_local[i] = shared_best[i];

            for (int warp = 0; warp < k_warp_count; ++warp) {
                for (int i = 0; i < warp_best_size[warp]; ++i) {
                    insert_topk<KStatic>(best_local, local_size, warp_best[warp][i]);
                }
            }

            shared_best_size = local_size;
            for (int i = 0; i < local_size; ++i)
                shared_best[i] = best_local[i];
        }

        __syncthreads();
    }

    const std::uint32_t kk = min(k, static_cast<std::uint32_t>(KStatic));
    const std::size_t out_base = static_cast<std::size_t>(blockIdx.x) * kk;

    for (std::uint32_t i = static_cast<std::uint32_t>(threadIdx.x); i < kk;
         i += static_cast<std::uint32_t>(blockDim.x)) {
        if (i < static_cast<std::uint32_t>(shared_best_size))
            block_results[out_base + i] = shared_best[i];
        else
            block_results[out_base + i] = DeviceTopKEntry{0, 0};
    }
}

std::size_t dynamic_shared_limit_bytes() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    return std::max<std::size_t>(static_cast<std::size_t>(prop.sharedMemPerBlock),
                                 static_cast<std::size_t>(prop.sharedMemPerBlockOptin));
}

template <int LowBits> std::size_t kernel_dynamic_shared_bytes() {
    return (std::size_t{1} << LowBits) * sizeof(std::int16_t);
}

template <int LowBits> std::size_t kernel_static_shared_bytes() {
    cudaFuncAttributes attr{};
    check_cuda(cudaFuncGetAttributes(&attr, sweep_persistent_kernel<LowBits>),
               "cudaFuncGetAttributes(sweep_persistent_kernel)");
    return static_cast<std::size_t>(attr.sharedSizeBytes);
}

HostGroupedPlan build_grouped_plan(const CompactSplitWindow& prep) {
    const std::size_t n_low = std::size_t{1} << prep.low_bits;
    const std::size_t unique_count = prep.multiplicity.size();

    HostGroupedPlan out;
    out.low_bits = prep.low_bits;
    out.high_bits = prep.high_bits;
    out.total_weight = prep.total_weight;
    out.r_offsets.assign(n_low + 1, 0);
    out.q.resize(unique_count);
    out.multiplicity.resize(unique_count);

    for (const std::uint64_t r64 : prep.r) {
        if (r64 >= n_low) {
            throw std::invalid_argument(
                "run_cuda_mitm_fwht_search: low-part pattern index is outside the selected split");
        }

        ++out.r_offsets[static_cast<std::size_t>(r64) + 1];
    }

    for (std::size_t i = 1; i < out.r_offsets.size(); ++i)
        out.r_offsets[i] += out.r_offsets[i - 1];

    std::vector<std::uint32_t> cursor = out.r_offsets;

    for (std::size_t i = 0; i < unique_count; ++i) {
        const std::int32_t m = prep.multiplicity[i];
        if (m <= 0 || m > static_cast<std::int32_t>(std::numeric_limits<std::uint16_t>::max())) {
            throw std::invalid_argument(
                "run_cuda_mitm_fwht_search: multiplicity must be in uint16 range");
        }

        const std::size_t r = static_cast<std::size_t>(prep.r[i]);
        const std::uint32_t pos = cursor[r]++;

        out.q[pos] = prep.q[i];
        out.multiplicity[pos] = static_cast<std::uint16_t>(m);
    }

    return out;
}

template <int LowBits> void configure_shared_memory_launch() {
    const std::size_t dynamic_shared_bytes = kernel_dynamic_shared_bytes<LowBits>();
    const std::size_t total_shared_bytes =
        dynamic_shared_bytes + kernel_static_shared_bytes<LowBits>();

    if (total_shared_bytes > dynamic_shared_limit_bytes()) {
        throw std::invalid_argument(
            "run_cuda_mitm_fwht_search: low_bits exceeds the device total shared-memory limit");
    }

    if (dynamic_shared_bytes <= static_cast<std::size_t>(48) * 1024)
        return;

    check_cuda(cudaFuncSetAttribute(sweep_persistent_kernel<LowBits>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamic_shared_bytes)),
               "cudaFuncSetAttribute(sweep_persistent_kernel)");
}

template <int LowBits>
std::size_t recommend_persistent_blocks(const std::uint64_t high_state_count,
                                        const std::size_t shared_bytes) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    int active_blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,
                                                             sweep_persistent_kernel<LowBits>,
                                                             k_threads_per_block,
                                                             static_cast<int>(shared_bytes)),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    const std::size_t blocks_per_sm = static_cast<std::size_t>(std::max(active_blocks_per_sm, 1));
    const std::size_t desired_blocks =
        static_cast<std::size_t>(prop.multiProcessorCount) * blocks_per_sm;

    return static_cast<std::size_t>(
        std::min<std::uint64_t>(high_state_count, std::max<std::size_t>(desired_blocks, 1)));
}

template <int LowBits>
void launch_sweep_kernel(const HostGroupedPlan& plan,
                         MitmDeviceWorkspace& workspace,
                         const std::size_t k,
                         const std::uint64_t high_state_count) {
    const std::size_t shared_bytes = kernel_dynamic_shared_bytes<LowBits>();

    configure_shared_memory_launch<LowBits>();
    const std::size_t persistent_blocks =
        recommend_persistent_blocks<LowBits>(high_state_count, shared_bytes);

    ensure_result_buffers(workspace.results, persistent_blocks * k, k);

    sweep_persistent_kernel<LowBits>
        <<<static_cast<unsigned int>(persistent_blocks), k_threads_per_block, shared_bytes>>>(
            workspace.d_r_offsets,
            workspace.d_q,
            workspace.d_multiplicity,
            high_state_count,
            plan.total_weight,
            static_cast<std::uint32_t>(k),
            workspace.results.d_block_results);
    check_cuda(cudaGetLastError(), "sweep_persistent_kernel launch");

    cuda_launch_detail::merge_topk_kernel<<<1, 1>>>(
        workspace.results.d_block_results, persistent_blocks, k, workspace.results.d_out_results);
    check_cuda(cudaGetLastError(), "merge_topk_kernel launch");
}

void dispatch_sweep_kernel(const HostGroupedPlan& plan,
                           MitmDeviceWorkspace& workspace,
                           const std::size_t k) {
    const std::uint64_t high_state_count = std::uint64_t{1} << plan.high_bits;

    switch (plan.low_bits) {
    case 1:
        launch_sweep_kernel<1>(plan, workspace, k, high_state_count);
        break;
    case 2:
        launch_sweep_kernel<2>(plan, workspace, k, high_state_count);
        break;
    case 3:
        launch_sweep_kernel<3>(plan, workspace, k, high_state_count);
        break;
    case 4:
        launch_sweep_kernel<4>(plan, workspace, k, high_state_count);
        break;
    case 5:
        launch_sweep_kernel<5>(plan, workspace, k, high_state_count);
        break;
    case 6:
        launch_sweep_kernel<6>(plan, workspace, k, high_state_count);
        break;
    case 7:
        launch_sweep_kernel<7>(plan, workspace, k, high_state_count);
        break;
    case 8:
        launch_sweep_kernel<8>(plan, workspace, k, high_state_count);
        break;
    case 9:
        launch_sweep_kernel<9>(plan, workspace, k, high_state_count);
        break;
    case 10:
        launch_sweep_kernel<10>(plan, workspace, k, high_state_count);
        break;
    case 11:
        launch_sweep_kernel<11>(plan, workspace, k, high_state_count);
        break;
    case 12:
        launch_sweep_kernel<12>(plan, workspace, k, high_state_count);
        break;
    case 13:
        launch_sweep_kernel<13>(plan, workspace, k, high_state_count);
        break;
    case 14:
        launch_sweep_kernel<14>(plan, workspace, k, high_state_count);
        break;
    case 15:
        launch_sweep_kernel<15>(plan, workspace, k, high_state_count);
        break;
    default:
        throw std::invalid_argument(
            "run_cuda_mitm_fwht_search: low_bits is outside the specialized CUDA range [1, 15]");
    }
}

MitmDeviceWorkspace& get_workspace() {
    thread_local MitmDeviceWorkspace workspace;
    int current_device = 0;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");

    if (workspace.device != current_device) {
        workspace.reset();
        workspace.device = current_device;
    }

    return workspace;
}

void upload_plan(const HostGroupedPlan& plan, MitmDeviceWorkspace& workspace) {
    ensure_buffer(workspace.d_r_offsets,
                  workspace.r_offsets_capacity,
                  plan.r_offsets.size(),
                  "cudaMalloc(r_offsets)");
    ensure_buffer(workspace.d_q, workspace.q_capacity, plan.q.size(), "cudaMalloc(q)");
    ensure_buffer(workspace.d_multiplicity,
                  workspace.multiplicity_capacity,
                  plan.multiplicity.size(),
                  "cudaMalloc(multiplicity)");

    check_cuda(cudaMemcpy(workspace.d_r_offsets,
                          plan.r_offsets.data(),
                          plan.r_offsets.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(r_offsets)");
    check_cuda(cudaMemcpy(workspace.d_q,
                          plan.q.data(),
                          plan.q.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(q)");
    check_cuda(cudaMemcpy(workspace.d_multiplicity,
                          plan.multiplicity.data(),
                          plan.multiplicity.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(multiplicity)");
}

} // namespace

std::vector<CudaMitmFwhtResult> run_cuda_mitm_fwht_search(const CompactSplitWindow& prep,
                                                          const std::size_t max_candidates) {
    if (max_candidates == 0 || prep.multiplicity.empty())
        return {};

    if (max_candidates > kMaxCandidates) {
        throw std::invalid_argument("run_cuda_mitm_fwht_search: max_candidates must be <= 128");
    }

    if (prep.low_bits == 0 || prep.low_bits > k_max_specialized_low_bits) {
        throw std::invalid_argument(
            "run_cuda_mitm_fwht_search: low_bits must be in the specialized CUDA range [1, 15]");
    }

    const HostGroupedPlan plan = build_grouped_plan(prep);
    MitmDeviceWorkspace& workspace = get_workspace();
    upload_plan(plan, workspace);

    dispatch_sweep_kernel(plan, workspace, max_candidates);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<DeviceTopKEntry> host_out(max_candidates);
    check_cuda(cudaMemcpy(host_out.data(),
                          workspace.results.d_out_results,
                          max_candidates * sizeof(DeviceTopKEntry),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(out_results)");

    std::vector<CudaMitmFwhtResult> out;
    out.reserve(max_candidates);
    for (const DeviceTopKEntry& entry : host_out) {
        if (entry.mask != 0)
            out.push_back(CudaMitmFwhtResult{entry.mask, entry.weight});
    }

    return out;
}

} // namespace bmmpy

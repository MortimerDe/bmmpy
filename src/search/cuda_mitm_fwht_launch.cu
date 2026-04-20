#include "bmmpy/search/cuda_mitm_fwht_launch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

constexpr int kMaxCandidates = 64;
constexpr int kThreadsPerBlock = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = kThreadsPerBlock / kWarpSize;
constexpr std::size_t kMaxSpecializedLowBits = 15;

static_assert(kThreadsPerBlock % kWarpSize == 0,
              "kThreadsPerBlock must be a multiple of warp size");

struct DeviceTopKEntry {
    std::uint64_t mask;
    std::uint32_t weight;
};

struct HostGroupedPlan {
    std::size_t low_bits = 0;
    std::size_t high_bits = 0;
    std::int32_t total_weight = 0;

    std::vector<std::uint32_t> r_offsets;
    std::vector<std::uint64_t> q;
    std::vector<std::uint16_t> multiplicity;
};

struct DeviceWorkspace {
    int device = -1;

    std::uint32_t* d_r_offsets = nullptr;
    std::uint64_t* d_q = nullptr;
    std::uint16_t* d_multiplicity = nullptr;

    DeviceTopKEntry* d_block_results = nullptr;
    DeviceTopKEntry* d_out_results = nullptr;

    std::size_t r_offsets_capacity = 0;
    std::size_t q_capacity = 0;
    std::size_t multiplicity_capacity = 0;
    std::size_t block_results_capacity = 0;
    std::size_t out_results_capacity = 0;

    ~DeviceWorkspace() { reset(); }

    void reset() noexcept {
        if (d_r_offsets != nullptr)
            (void)cudaFree(d_r_offsets);
        if (d_q != nullptr)
            (void)cudaFree(d_q);
        if (d_multiplicity != nullptr)
            (void)cudaFree(d_multiplicity);
        if (d_block_results != nullptr)
            (void)cudaFree(d_block_results);
        if (d_out_results != nullptr)
            (void)cudaFree(d_out_results);

        d_r_offsets = nullptr;
        d_q = nullptr;
        d_multiplicity = nullptr;
        d_block_results = nullptr;
        d_out_results = nullptr;

        r_offsets_capacity = 0;
        q_capacity = 0;
        multiplicity_capacity = 0;
        block_results_capacity = 0;
        out_results_capacity = 0;
        device = -1;
    }
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
    __shared__ DeviceTopKEntry warp_best[kWarpCount][KStatic];
    __shared__ int warp_best_size[kWarpCount];
    __shared__ DeviceTopKEntry warp_stage[kWarpCount][kWarpSize];

    const int warp_id = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane_id = static_cast<int>(threadIdx.x) % kWarpSize;

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

        for (std::uint32_t y_base = static_cast<std::uint32_t>(warp_id * kWarpSize); y_base < n_low;
             y_base += static_cast<std::uint32_t>(kWarpCount * kWarpSize)) {
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
                for (int lane = 0; lane < kWarpSize; ++lane) {
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

            for (int warp = 0; warp < kWarpCount; ++warp) {
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

__global__ void merge_kernel(const DeviceTopKEntry* block_results,
                             const std::size_t block_count,
                             const std::size_t k,
                             DeviceTopKEntry* out_results) {
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    constexpr int KStatic = kMaxCandidates;
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

void check_cuda(const cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + context + ": " +
                                 cudaGetErrorString(status));
    }
}

std::size_t next_capacity(const std::size_t current, const std::size_t required) {
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
void ensure_buffer(T*& ptr, std::size_t& capacity, const std::size_t required, const char* label) {
    if (required <= capacity)
        return;

    if (ptr != nullptr)
        check_cuda(cudaFree(ptr), "cudaFree(realloc)");

    const std::size_t new_capacity = next_capacity(capacity, required);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), new_capacity * sizeof(T)), label);
    capacity = new_capacity;
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
                                                             kThreadsPerBlock,
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
                         DeviceWorkspace& workspace,
                         const std::size_t k,
                         const std::uint64_t high_state_count) {
    const std::size_t shared_bytes = kernel_dynamic_shared_bytes<LowBits>();

    configure_shared_memory_launch<LowBits>();
    const std::size_t persistent_blocks =
        recommend_persistent_blocks<LowBits>(high_state_count, shared_bytes);

    ensure_buffer(workspace.d_block_results,
                  workspace.block_results_capacity,
                  persistent_blocks * k,
                  "cudaMalloc(block_results)");
    ensure_buffer(
        workspace.d_out_results, workspace.out_results_capacity, k, "cudaMalloc(out_results)");

    sweep_persistent_kernel<LowBits>
        <<<static_cast<unsigned int>(persistent_blocks), kThreadsPerBlock, shared_bytes>>>(
            workspace.d_r_offsets,
            workspace.d_q,
            workspace.d_multiplicity,
            high_state_count,
            plan.total_weight,
            static_cast<std::uint32_t>(k),
            workspace.d_block_results);
    check_cuda(cudaGetLastError(), "sweep_persistent_kernel launch");

    merge_kernel<<<1, 1>>>(
        workspace.d_block_results, persistent_blocks, k, workspace.d_out_results);
    check_cuda(cudaGetLastError(), "merge_kernel launch");
}

void dispatch_sweep_kernel(const HostGroupedPlan& plan,
                           DeviceWorkspace& workspace,
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

DeviceWorkspace& get_workspace() {
    thread_local DeviceWorkspace workspace;
    int current_device = 0;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");

    if (workspace.device != current_device) {
        workspace.reset();
        workspace.device = current_device;
    }

    return workspace;
}

void upload_plan(const HostGroupedPlan& plan, DeviceWorkspace& workspace) {
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
        throw std::invalid_argument("run_cuda_mitm_fwht_search: max_candidates must be <= 64");
    }

    if (prep.low_bits == 0 || prep.low_bits > kMaxSpecializedLowBits) {
        throw std::invalid_argument(
            "run_cuda_mitm_fwht_search: low_bits must be in the specialized CUDA range [1, 15]");
    }

    const HostGroupedPlan plan = build_grouped_plan(prep);
    DeviceWorkspace& workspace = get_workspace();
    upload_plan(plan, workspace);

    dispatch_sweep_kernel(plan, workspace, max_candidates);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<DeviceTopKEntry> host_out(max_candidates);
    check_cuda(cudaMemcpy(host_out.data(),
                          workspace.d_out_results,
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
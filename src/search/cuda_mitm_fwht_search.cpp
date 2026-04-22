#include "bmmpy/search/cuda_mitm_fwht_search.hpp"

#include "bmmpy/search/cuda_mitm_fwht_launch.hpp"
#include "bmmpy/search/split_window_prep.hpp"
#include "bmmpy/stub.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#if defined(BMMPY_HAS_CUDA_BACKEND)
#include <cuda_runtime_api.h>
#endif

namespace bmmpy {
namespace {

constexpr std::size_t kFallbackAutoLowBits = 10;
constexpr std::size_t kMaxSpecializedLowBits = 15;
constexpr std::size_t kPreferredMinLowBits = 8;
constexpr double kExpectedUniquePatterns = 512.0;
constexpr std::size_t kConservativeStaticSharedReserve = 16 * 1024;

std::size_t floor_log2_size(std::size_t value) noexcept {
    std::size_t out = 0;
    while ((std::size_t{1} << (out + 1)) <= value)
        ++out;
    return out;
}

std::size_t detect_cuda_shared_limited_low_bits() noexcept {
#if !defined(BMMPY_HAS_CUDA_BACKEND)
    return kFallbackAutoLowBits;
#else
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return kFallbackAutoLowBits;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return kFallbackAutoLowBits;

    const std::size_t shared_limit =
        std::max<std::size_t>(static_cast<std::size_t>(prop.sharedMemPerBlock),
                              static_cast<std::size_t>(prop.sharedMemPerBlockOptin));

    if (shared_limit <= kConservativeStaticSharedReserve + sizeof(std::int16_t))
        return 1;

    const std::size_t dynamic_budget = shared_limit - kConservativeStaticSharedReserve;
    std::size_t low_bits = 0;
    while (low_bits + 1 < Candidate::k_word_bits &&
           ((std::size_t{1} << (low_bits + 1)) * sizeof(std::int16_t)) <= dynamic_budget) {
        ++low_bits;
    }

    return std::min<std::size_t>(low_bits, kMaxSpecializedLowBits);
#endif
}

std::size_t detect_target_persistent_blocks() noexcept {
#if !defined(BMMPY_HAS_CUDA_BACKEND)
    return 128;
#else
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 128;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return 128;

    return static_cast<std::size_t>(std::max(prop.multiProcessorCount, 1)) * 8;
#endif
}

std::size_t resolve_low_bits(const std::size_t t, const std::size_t configured_low_bits) noexcept {
    const std::size_t max_low_bits =
        std::min<std::size_t>(t - 1, detect_cuda_shared_limited_low_bits());

    if (configured_low_bits != 0)
        return configured_low_bits;

    if (max_low_bits == 0)
        return 0;

    const std::size_t target_blocks = detect_target_persistent_blocks();
    const std::size_t min_high_bits_for_occupancy =
        target_blocks <= 1 ? 0 : floor_log2_size(target_blocks - 1) + 1;

    std::size_t best_low_bits = std::min<std::size_t>(max_low_bits, kFallbackAutoLowBits);
    double best_cost = std::numeric_limits<double>::infinity();

    for (std::size_t low_bits = 1; low_bits <= max_low_bits; ++low_bits) {
        if (max_low_bits >= kPreferredMinLowBits && low_bits < kPreferredMinLowBits)
            continue;

        const std::size_t high_bits = t - low_bits;
        const double build_cost =
            kExpectedUniquePatterns / static_cast<double>(std::size_t{1} << low_bits);
        const double fwht_cost = static_cast<double>(low_bits);

        double occupancy_penalty = 0.0;
        if (high_bits < min_high_bits_for_occupancy) {
            occupancy_penalty = 32.0 * static_cast<double>(min_high_bits_for_occupancy - high_bits);
        }

        const double total_cost = build_cost + fwht_cost + occupancy_penalty;
        if (total_cost < best_cost) {
            best_cost = total_cost;
            best_low_bits = low_bits;
        }
    }

    return best_low_bits;
}

constexpr std::int32_t kMaxInt16SafeTotalWeight =
    static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::max());

} // namespace

std::vector<Candidate> CudaMitmFwhtSearch::search(const RowWindow& window) {
    if (_config.max_candidates == 0)
        return {};

    const std::size_t t = window.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: too many rows in the window (must be <= 64)");
    }

    if (t < kMinRows || t > kMaxRows) {
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: window_rows size is outside the supported GPU range");
    }

    const std::size_t low_bits = resolve_low_bits(t, _config.low_bits);
    if (low_bits == 0 || low_bits >= t) {
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: low_bits must be in [1, t - 1] or 0 for auto");
    }

    if (low_bits > kMaxSpecializedLowBits) {
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: low_bits exceeds the specialized CUDA template range");
    }

    if (window.words_per_row() == 0)
        return {};

    const CompactSplitWindow prep = build_compact_split_window(window, low_bits);
    if (prep.multiplicity.empty())
        return {};

    if (prep.total_weight > kMaxInt16SafeTotalWeight) {
        throw std::invalid_argument(
            "CudaMitmFwhtSearch: total_weight must be <= 32767 for int16-safe GPU mode");
    }

#if !defined(BMMPY_HAS_CUDA_BACKEND)
    throw std::runtime_error("CudaMitmFwhtSearch: CUDA backend is not compiled into this build");
#else
    const RuntimeFeatures features = get_runtime_features();
    if (!features.cuda_available)
        throw std::runtime_error("CudaMitmFwhtSearch: no CUDA device is available");

    const auto device_results = run_cuda_mitm_fwht_search(prep, _config.max_candidates);

    std::vector<Candidate> out;
    out.reserve(device_results.size());
    for (const auto& entry : device_results)
        out.push_back(Candidate::from_u64(entry.mask, entry.weight));

    return out;
#endif
}

} // namespace bmmpy
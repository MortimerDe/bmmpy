#include "bmmpy/search/cuda_mitm_fwht_search.hpp"

#include "bmmpy/search/split_window_prep.hpp"
#include "bmmpy/stub.hpp"

#include <limits>
#include <stdexcept>

namespace bmmpy {
namespace {

constexpr std::size_t kAutoLowBits = 13;
constexpr std::int32_t kMaxInt16SafeTotalWeight =
    static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::max());

std::size_t resolve_low_bits(std::size_t t, std::size_t configured_low_bits) noexcept {
    if (configured_low_bits != 0)
        return configured_low_bits;

    return std::min<std::size_t>(kAutoLowBits, t / 2);
}

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

    (void)prep;
    throw std::runtime_error("CudaMitmFwhtSearch: CUDA search path is not implemented yet");
#endif
}

} // namespace bmmpy
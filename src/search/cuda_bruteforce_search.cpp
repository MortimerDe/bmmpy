#include "bmmpy/search/cuda_bruteforce_search.hpp"

#include "bmmpy/search/cuda_bruteforce_launch.hpp"
#include "bmmpy/stub.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

constexpr std::size_t kMaxCudaCandidates = 128;
constexpr std::size_t kSupportedWordsPerRow512 = 8;
constexpr std::size_t kSupportedWordsPerRow1024 = 16;

std::size_t resolve_chunk_bits(const std::size_t t, const std::size_t configured_chunk_bits) {
    if (configured_chunk_bits == 0)
        return 0;

    if (configured_chunk_bits >= t || configured_chunk_bits >= Candidate::k_word_bits) {
        throw std::invalid_argument(
            "CudaBruteforceSearch: chunk_bits must be in [1, t - 1] or 0 for auto");
    }

    return configured_chunk_bits;
}

bool is_supported_words_per_row(const std::size_t words_per_row) noexcept {
    return words_per_row == kSupportedWordsPerRow512 || words_per_row == kSupportedWordsPerRow1024;
}

} // namespace

std::vector<Candidate> CudaBruteforceSearch::search(const RowWindow& window) {
    if (_config.max_candidates == 0)
        return {};

    const std::size_t t = window.size();
    if (t == 0)
        return {};

    if (t > Candidate::k_word_bits) {
        throw std::invalid_argument(
            "CudaBruteforceSearch: too many rows in the window (must be <= 64)");
    }

    if (t < kMinRows || t > kMaxRows) {
        throw std::invalid_argument(
            "CudaBruteforceSearch: window_rows size is outside the supported GPU range");
    }

    if (_config.max_candidates > kMaxCudaCandidates) {
        throw std::invalid_argument("CudaBruteforceSearch: max_candidates must be <= 128");
    }

    const std::size_t words_per_row = window.words_per_row();
    if (words_per_row == 0)
        return {};

    if (!is_supported_words_per_row(words_per_row)) {
        throw std::invalid_argument(
            "CudaBruteforceSearch: only widths 512 and 1024 are supported in the current GPU path");
    }

    const std::size_t chunk_bits = resolve_chunk_bits(t, _config.chunk_bits);

#if !defined(BMMPY_HAS_CUDA_BACKEND)
    (void)chunk_bits;
    throw std::runtime_error("CudaBruteforceSearch: CUDA backend is not compiled into this build");
#else
    const RuntimeFeatures features = get_runtime_features();
    if (!features.cuda_available)
        throw std::runtime_error("CudaBruteforceSearch: no CUDA device is available");

    CudaBruteforcePlan plan;
    plan.rows = t;
    plan.cols = window.cols();
    plan.words_per_row = words_per_row;
    plan.chunk_bits = chunk_bits;

    const auto& rows = window.row_ptrs();
    plan.row_words.resize(t * words_per_row);

    for (std::size_t row = 0; row < t; ++row) {
        std::copy_n(rows[row],
                    words_per_row,
                    plan.row_words.begin() + static_cast<std::ptrdiff_t>(row * words_per_row));
    }

    const auto device_results = run_cuda_bruteforce_search(plan, _config.max_candidates);

    std::vector<Candidate> out;
    out.reserve(device_results.size());
    for (const auto& entry : device_results)
        out.push_back(Candidate::from_u64(entry.mask, entry.weight));

    return out;
#endif
}

} // namespace bmmpy
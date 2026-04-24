#include "bmmpy/search/cuda_bruteforce_launch.hpp"
#include "bmmpy/search/cuda_launch_topk.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

using cuda_launch_detail::kMaxCandidates;

constexpr std::size_t kAutoChunkBits = 12;
constexpr std::size_t kSupportedWordsPerRow512 = 8;
constexpr std::size_t kSupportedWordsPerRow1024 = 16;
constexpr std::size_t kMaxRows = 64;
constexpr std::size_t kMaskBits = 64;

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

    (void)chunk_bits;
    (void)prefix_count;

    return {};
}

} // namespace bmmpy
#include "bmmpy/search/cuda_bruteforce_launch.hpp"
#include "cuda_launch_result_buffers.cuh"
#include "cuda_launch_runtime.cuh"
#include "cuda_launch_topk.cuh"

#include <limits>
#include <stdexcept>
#include <vector>

namespace bmmpy {
namespace {

using cuda_launch_detail::kMaxCandidates;

constexpr std::size_t kSupportedWordsPerRow512 = 8;
constexpr std::size_t kSupportedWordsPerRow1024 = 16;
constexpr std::size_t kMaxRows = 64;

bool is_supported_words_per_row(const std::size_t words_per_row) noexcept {
    return words_per_row == kSupportedWordsPerRow512 || words_per_row == kSupportedWordsPerRow1024;
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

    throw std::runtime_error(
        "run_cuda_bruteforce_search: CUDA brute-force backend is not implemented yet");
}

} // namespace bmmpy
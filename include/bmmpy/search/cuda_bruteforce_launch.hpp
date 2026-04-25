#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

struct CudaBruteforcePlan {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t words_per_row = 0;
    std::size_t chunk_bits = 0; // 0 = auto
    std::vector<std::uint64_t> row_words;
};

struct CudaBruteforceResult {
    std::uint64_t mask;
    std::uint32_t weight;
};

std::vector<CudaBruteforceResult> run_cuda_bruteforce_search(const CudaBruteforcePlan& plan,
                                                             std::size_t max_candidates);

} // namespace bmmpy
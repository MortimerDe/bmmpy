#pragma once

#include "bmmpy/search/split_window_prep.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {
struct CudaMitmFwhtResult {
    std::uint64_t mask;
    std::uint32_t weight;
};

std::vector<CudaMitmFwhtResult> run_cuda_mitm_fwht_search(const CompactSplitWindow& prep,
                                                          std::size_t max_candidates);
} // namespace bmmpy
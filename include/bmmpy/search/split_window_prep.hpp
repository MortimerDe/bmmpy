#pragma once

#include "bmmpy/core/row_window.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {
struct CompactSplitWindow { // compact repr of a row window
    std::size_t t = 0;
    std::size_t low_bits = 0;
    std::size_t high_bits = 0;
    std::int32_t total_weight = 0;

    std::vector<std::uint64_t> q;
    std::vector<std::uint64_t> r;
    std::vector<std::int32_t> multiplicity;
};

CompactSplitWindow build_compact_split_window(const RowWindow& window, std::size_t low_bits);

CompactSplitWindow build_compact_split_window(const std::vector<const std::uint64_t*>& rows,
                                              std::size_t words_per_row,
                                              std::size_t cols,
                                              std::size_t low_bits);
} // namespace bmmpy
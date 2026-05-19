#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/xorshift64.hpp"
#include "bmmpy/search/sa_selector.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace bmmpy::sa_detail {

constexpr std::size_t k_word_bits = std::numeric_limits<std::uint64_t>::digits;

struct RestartResult {
    std::vector<std::size_t> rows;
    std::int64_t score = std::numeric_limits<std::int64_t>::min();
    std::size_t accepted_moves = 0;
    std::size_t iterations_run = 0;
    std::size_t best_iteration = 0;
};

inline bool is_valid_probability(double value) noexcept { return value > 0.0 && value < 1.0; }

inline bool is_better_result(std::int64_t lhs_score,
                             const std::vector<std::size_t>& lhs_rows,
                             std::int64_t rhs_score,
                             const std::vector<std::size_t>& rhs_rows) {
    if (lhs_score != rhs_score)
        return lhs_score > rhs_score;
    return lhs_rows < rhs_rows;
}

inline std::vector<std::size_t> sorted_copy(const std::vector<std::size_t>& rows) {
    std::vector<std::size_t> out = rows;
    std::sort(out.begin(), out.end());
    return out;
}

inline std::uint64_t tail_mask_for_cols(std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};
    return (std::uint64_t{1} << tail_bits) - 1;
}

inline void initialize_partition(std::size_t row_count,
                                 std::size_t window_size,
                                 detail::XorShift64& rng,
                                 std::vector<std::size_t>& window,
                                 std::vector<std::size_t>& pool) {
    pool.resize(row_count);
    std::iota(pool.begin(), pool.end(), 0);

    for (std::size_t i = 0; i < window_size; ++i) {
        const std::size_t j = i + rng.next_index(row_count - i);
        std::swap(pool[i], pool[j]);
    }

    window.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(window_size));
    pool.erase(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(window_size));
}

RestartResult run_pairwise_restart(const BitMatrix& matrix,
                                   std::size_t window_size,
                                   const SASelectorConfig& config,
                                   detail::XorShift64& rng);

RestartResult run_higher_order_restart(const BitMatrix& matrix,
                                       std::size_t window_size,
                                       const SASelectorConfig& config,
                                       detail::XorShift64& rng);

} // namespace bmmpy::sa_detail
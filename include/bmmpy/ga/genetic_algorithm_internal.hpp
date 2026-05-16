#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/ga/genetic_algorithm.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace bmmpy::ga::internal {

using steady_clock = std::chrono::steady_clock;

inline long long elapsed_ms(const steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - started)
        .count();
}

inline bool
should_stop(const GeneticAlgorithmConfig& cfg, const RunStats& st, const std::size_t best_score) {
    if (cfg.stop.max_generations && st.generations >= *cfg.stop.max_generations)
        return true;
    if (cfg.stop.max_stale_generations && st.stale_generations >= *cfg.stop.max_stale_generations)
        return true;
    if (cfg.stop.target_total_weight && best_score <= *cfg.stop.target_total_weight)
        return true;
    return false;
}

inline std::uint64_t tail_mask_for_cols(const std::size_t cols) noexcept {
    const std::size_t tail_bits = cols % Candidate::k_word_bits;
    if (tail_bits == 0)
        return ~std::uint64_t{0};

    return (std::uint64_t{1} << tail_bits) - 1;
}

inline void mat_cand(const RowWindow& window,
                     const std::size_t row_count,
                     const Candidate& candidate,
                     std::uint64_t* out_words) {
    const std::size_t word_count = window.words_per_row();
    if (word_count == 0)
        return;

    if (out_words == nullptr) {
        throw std::invalid_argument("mat_cand: out_words must not be null when word_count > 0");
    }

    std::fill(out_words, out_words + word_count, std::uint64_t{0});

    const auto& ops = ::bmmpy::detail::bit_ops();
    for (auto r : candidate.selected_rows()) {
        if (r >= row_count) {
            throw std::out_of_range("mat_cand: candidate row index exceeds window size");
        }

        ops.row_xor(out_words, window.row_words(r), word_count);
    }

    out_words[word_count - 1] &= tail_mask_for_cols(window.cols());
}

inline std::uint32_t eval_mat_cand(const std::uint64_t* row_words,
                                   const std::size_t word_count,
                                   const std::uint64_t tail_mask) {
    if (word_count == 0)
        return 0;

    const auto& ops = ::bmmpy::detail::bit_ops();

    std::uint64_t total = 0;
    if (word_count > 1)
        total += ops.row_popcount(row_words, word_count - 1);

    const std::uint64_t tail_word = row_words[word_count - 1] & tail_mask;
    total += ops.row_popcount(&tail_word, 1);

    return static_cast<std::uint32_t>(total);
}

inline std::uint32_t eval_cand_weight(const RowWindow& window,
                                      const std::size_t row_count,
                                      const std::size_t col_count,
                                      const Candidate& candidate,
                                      std::uint64_t* scratch_words) {
    mat_cand(window, row_count, candidate, scratch_words);
    return eval_mat_cand(scratch_words, window.words_per_row(), tail_mask_for_cols(col_count));
}

inline std::uint32_t eval_cand_weight(const RowWindow& window,
                                      const std::size_t row_count,
                                      const std::size_t col_count,
                                      const Candidate& candidate) {
    if (col_count == 0)
        return 0;

    ::bmmpy::BitMatrix scratch(1, col_count);
    return eval_cand_weight(window, row_count, col_count, candidate, scratch.row_words(0));
}

} // namespace bmmpy::ga::internal

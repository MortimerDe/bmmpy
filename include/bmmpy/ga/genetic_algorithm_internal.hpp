#pragma once

#include "bmmpy/ga/genetic_algorithm.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

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

inline std::uint32_t eval_cand_weight(const RowWindow& window,
                                      const std::size_t row_count,
                                      const std::size_t col_count,
                                      const Candidate& candidate) {
    std::vector<bool> row(col_count, false);

    // for (std::size_t r = 0; r < row_count; ++r) {
    //     if (!candidate.has_row(r))
    //         continue;

    //     for (std::size_t col = 0; col < col_count; ++col)
    //         row[col] = row[col] != window.get(r, col);
    // }

    for (auto r : candidate.selected_rows()) {
        for (std::size_t col = 0; col < col_count; ++col)
            row[col] = row[col] != window.get(r, col);
    }

    return static_cast<std::uint32_t>(std::count(row.begin(), row.end(), true));
}

} // namespace bmmpy::ga::internal
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/core/detail/xorshift64.hpp"
#include "bmmpy/search/sa_selector_internal.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace bmmpy::sa_detail {
namespace {

struct PairwiseSynergyTable {
    std::size_t rows = 0;
    std::vector<std::uint32_t> values;

    std::uint32_t at(std::size_t i, std::size_t j) const noexcept { return values[i * rows + j]; }
    std::uint32_t& at(std::size_t i, std::size_t j) noexcept { return values[i * rows + j]; }
};

PairwiseSynergyTable build_pairwise_synergy_table(const BitMatrix& matrix) {
    const std::size_t row_count = matrix.rows();
    if (row_count != 0 && row_count > (std::numeric_limits<std::size_t>::max() / row_count)) {
        throw std::overflow_error("SASelector: synergy matrix size overflow");
    }

    PairwiseSynergyTable table;
    table.rows = row_count;
    table.values.assign(row_count * row_count, 0);

    const auto& ops = detail::bit_ops();
    const std::size_t words_per_row = matrix.words_per_row();

    for (std::size_t i = 0; i < row_count; ++i) {
        const std::uint64_t* row_i = matrix.row_words(i);
        for (std::size_t j = i + 1; j < row_count; ++j) {
            const std::uint64_t overlap =
                ops.row_and_popcount(row_i, matrix.row_words(j), words_per_row);

            if (overlap >
                static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max() / 2)) {
                throw std::overflow_error("SASelector: pairwise synergy exceeds uint32 range");
            }

            const std::uint32_t synergy = static_cast<std::uint32_t>(overlap * 2);
            table.at(i, j) = synergy;
            table.at(j, i) = synergy;
        }
    }

    return table;
}

std::int64_t compute_window_score(const std::vector<std::size_t>& window,
                                  const PairwiseSynergyTable& table) {
    std::int64_t score = 0;
    for (std::size_t i = 0; i < window.size(); ++i) {
        for (std::size_t j = i + 1; j < window.size(); ++j)
            score += static_cast<std::int64_t>(table.at(window[i], window[j]));
    }
    return score;
}

std::int64_t compute_swap_delta(const std::vector<std::size_t>& window,
                                std::size_t outgoing_index,
                                std::size_t incoming_row,
                                const PairwiseSynergyTable& table) {
    const std::size_t outgoing_row = window[outgoing_index];

    std::int64_t delta = 0;
    for (std::size_t i = 0; i < window.size(); ++i) {
        if (i == outgoing_index)
            continue;

        const std::size_t neighbor = window[i];
        delta += static_cast<std::int64_t>(table.at(incoming_row, neighbor));
        delta -= static_cast<std::int64_t>(table.at(outgoing_row, neighbor));
    }

    return delta;
}

double estimate_initial_temperature(const std::vector<std::size_t>& window,
                                    const std::vector<std::size_t>& pool,
                                    const PairwiseSynergyTable& table,
                                    const SASelectorConfig& config,
                                    detail::XorShift64& rng) {
    if (pool.empty())
        return 1.0;

    const std::size_t sample_count = std::max<std::size_t>(config.temperature_probe_samples, 1);
    double negative_delta_sum = 0.0;
    std::size_t negative_count = 0;

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::size_t idx_w = rng.next_index(window.size());
        const std::size_t idx_p = rng.next_index(pool.size());

        const std::int64_t delta = compute_swap_delta(window, idx_w, pool[idx_p], table);
        if (delta < 0) {
            negative_delta_sum += static_cast<double>(-delta);
            ++negative_count;
        }
    }

    if (negative_count == 0)
        return 1.0;

    const double avg_negative_delta = negative_delta_sum / static_cast<double>(negative_count);
    const double temperature =
        -avg_negative_delta / std::log(config.initial_acceptance_probability);
    return std::max(temperature, config.min_temperature);
}

} // namespace

RestartResult run_pairwise_restart(const BitMatrix& matrix,
                                   std::size_t window_size,
                                   const SASelectorConfig& config,
                                   detail::XorShift64& rng) {
    const PairwiseSynergyTable table = build_pairwise_synergy_table(matrix);

    std::vector<std::size_t> window;
    std::vector<std::size_t> pool;
    initialize_partition(table.rows, window_size, rng, window, pool);

    std::int64_t score = compute_window_score(window, table);

    RestartResult best;
    best.rows = sorted_copy(window);
    best.score = score;
    best.iterations_run = pool.empty() ? 0 : config.iterations;

    if (pool.empty() || config.iterations == 0)
        return best;

    double temperature = estimate_initial_temperature(window, pool, table, config, rng);
    std::size_t accepted_moves = 0;

    for (std::size_t iter = 0; iter < config.iterations; ++iter) {
        const std::size_t idx_w = rng.next_index(window.size());
        const std::size_t idx_p = rng.next_index(pool.size());

        const std::size_t incoming = pool[idx_p];
        const std::int64_t delta = compute_swap_delta(window, idx_w, incoming, table);

        bool accept = delta > 0;
        if (!accept && temperature > config.min_temperature) {
            const double acceptance = std::exp(static_cast<double>(delta) / temperature);
            accept = rng.next_unit_double() < acceptance;
        }

        if (accept) {
            std::swap(window[idx_w], pool[idx_p]);
            score += delta;
            ++accepted_moves;

            const std::vector<std::size_t> sorted_rows = sorted_copy(window);
            if (is_better_result(score, sorted_rows, best.score, best.rows)) {
                best.rows = sorted_rows;
                best.score = score;
                best.accepted_moves = accepted_moves;
                best.best_iteration = iter + 1;
            }
        }

        temperature = std::max(config.min_temperature, temperature * config.cooling_rate);
    }

    return best;
}

} // namespace bmmpy::sa_detail
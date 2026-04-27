#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/search/sa_selector_internal.hpp"

#include <cmath>
#include <vector>

namespace bmmpy::sa_detail {
namespace {

template <typename Fn>
void for_each_set_bit(const std::uint64_t* row_words,
                      std::size_t words_per_row,
                      std::size_t cols,
                      Fn&& fn) {
    const std::uint64_t tail_mask = tail_mask_for_cols(cols);

    for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
        std::uint64_t word = row_words[word_index];
        if (word_index + 1 == words_per_row)
            word &= tail_mask;

        while (word != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(word));
            fn(word_index * kWordBits + bit);
            word &= (word - 1);
        }
    }
}

std::int64_t higher_order_score_term(std::uint32_t count) noexcept {
    const std::int64_t n = static_cast<std::int64_t>(count);
    return n * (n - 1) * (n - 1);
}

std::int64_t compute_window_score(const BitMatrix& matrix,
                                  const std::vector<std::size_t>& window,
                                  std::vector<std::uint32_t>& column_counts) {
    column_counts.assign(matrix.cols(), 0);

    const std::size_t words_per_row = matrix.words_per_row();
    for (std::size_t row : window) {
        for_each_set_bit(matrix.row_words(row), words_per_row, matrix.cols(), [&](std::size_t col) {
            ++column_counts[col];
        });
    }

    std::int64_t score = 0;
    for (std::uint32_t count : column_counts)
        score += higher_order_score_term(count);

    return score;
}

std::int64_t compute_swap_delta(const BitMatrix& matrix,
                                std::size_t outgoing_row,
                                std::size_t incoming_row,
                                const std::vector<std::uint32_t>& column_counts) {
    const std::size_t words_per_row = matrix.words_per_row();
    const std::uint64_t tail_mask = tail_mask_for_cols(matrix.cols());
    const std::uint64_t* outgoing = matrix.row_words(outgoing_row);
    const std::uint64_t* incoming = matrix.row_words(incoming_row);

    std::int64_t delta = 0;

    for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
        std::uint64_t out_word = outgoing[word_index];
        std::uint64_t in_word = incoming[word_index];

        if (word_index + 1 == words_per_row) {
            out_word &= tail_mask;
            in_word &= tail_mask;
        }

        std::uint64_t removed = out_word & ~in_word;
        while (removed != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(removed));
            const std::size_t col = word_index * kWordBits + bit;
            const std::uint32_t n = column_counts[col];
            delta += higher_order_score_term(n - 1) - higher_order_score_term(n);
            removed &= (removed - 1);
        }

        std::uint64_t added = in_word & ~out_word;
        while (added != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(added));
            const std::size_t col = word_index * kWordBits + bit;
            const std::uint32_t n = column_counts[col];
            delta += higher_order_score_term(n + 1) - higher_order_score_term(n);
            added &= (added - 1);
        }
    }

    return delta;
}

void apply_swap(const BitMatrix& matrix,
                std::size_t outgoing_row,
                std::size_t incoming_row,
                std::vector<std::uint32_t>& column_counts) {
    const std::size_t words_per_row = matrix.words_per_row();
    const std::uint64_t tail_mask = tail_mask_for_cols(matrix.cols());
    const std::uint64_t* outgoing = matrix.row_words(outgoing_row);
    const std::uint64_t* incoming = matrix.row_words(incoming_row);

    for (std::size_t word_index = 0; word_index < words_per_row; ++word_index) {
        std::uint64_t out_word = outgoing[word_index];
        std::uint64_t in_word = incoming[word_index];

        if (word_index + 1 == words_per_row) {
            out_word &= tail_mask;
            in_word &= tail_mask;
        }

        std::uint64_t removed = out_word & ~in_word;
        while (removed != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(removed));
            const std::size_t col = word_index * kWordBits + bit;
            --column_counts[col];
            removed &= (removed - 1);
        }

        std::uint64_t added = in_word & ~out_word;
        while (added != 0) {
            const std::size_t bit = static_cast<std::size_t>(detail::ctz64(added));
            const std::size_t col = word_index * kWordBits + bit;
            ++column_counts[col];
            added &= (added - 1);
        }
    }
}

double estimate_initial_temperature(const BitMatrix& matrix,
                                    const std::vector<std::size_t>& window,
                                    const std::vector<std::size_t>& pool,
                                    const std::vector<std::uint32_t>& column_counts,
                                    const SASelectorConfig& config,
                                    XorShift64& rng) {
    if (pool.empty())
        return 1.0;

    const std::size_t sample_count = std::max<std::size_t>(config.temperature_probe_samples, 1);
    double negative_delta_sum = 0.0;
    std::size_t negative_count = 0;

    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::size_t idx_w = rng.next_index(window.size());
        const std::size_t idx_p = rng.next_index(pool.size());

        const std::int64_t delta =
            compute_swap_delta(matrix, window[idx_w], pool[idx_p], column_counts);

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

RestartResult run_higher_order_restart(const BitMatrix& matrix,
                                       std::size_t window_size,
                                       const SASelectorConfig& config,
                                       XorShift64& rng) {
    std::vector<std::size_t> window;
    std::vector<std::size_t> pool;
    initialize_partition(matrix.rows(), window_size, rng, window, pool);

    std::vector<std::uint32_t> column_counts;
    std::int64_t score = compute_window_score(matrix, window, column_counts);

    RestartResult best;
    best.rows = sorted_copy(window);
    best.score = score;
    best.iterations_run = pool.empty() ? 0 : config.iterations;

    if (pool.empty() || config.iterations == 0)
        return best;

    double temperature =
        estimate_initial_temperature(matrix, window, pool, column_counts, config, rng);
    std::size_t accepted_moves = 0;

    for (std::size_t iter = 0; iter < config.iterations; ++iter) {
        const std::size_t idx_w = rng.next_index(window.size());
        const std::size_t idx_p = rng.next_index(pool.size());

        const std::size_t outgoing = window[idx_w];
        const std::size_t incoming = pool[idx_p];
        const std::int64_t delta = compute_swap_delta(matrix, outgoing, incoming, column_counts);

        bool accept = delta > 0;
        if (!accept && temperature > config.min_temperature) {
            const double acceptance = std::exp(static_cast<double>(delta) / temperature);
            accept = rng.next_unit_double() < acceptance;
        }

        if (accept) {
            apply_swap(matrix, outgoing, incoming, column_counts);
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
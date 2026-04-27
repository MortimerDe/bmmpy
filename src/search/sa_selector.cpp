#include "bmmpy/search/sa_selector.hpp"

#include "bmmpy/core/detail/bit_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy {
namespace {

constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ull;

class XorShift64 {
public:
    explicit XorShift64(std::uint64_t seed) noexcept : _state(seed == 0 ? kDefaultSeed : seed) {}

    std::uint64_t next_u64() noexcept {
        std::uint64_t x = _state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        _state = x;
        return x;
    }

    std::size_t next_index(std::size_t upper_bound) noexcept {
        if (upper_bound == 0)
            return 0;
        return static_cast<std::size_t>(next_u64() % upper_bound);
    }

    double next_unit_double() noexcept {
        constexpr double kInv53 = 1.0 / 9007199254740992.0;
        return static_cast<double>(next_u64() >> 11) * kInv53;
    }

private:
    std::uint64_t _state;
};

struct PairwiseSynergyTable {
    std::size_t rows = 0;
    std::vector<std::uint32_t> values;

    std::uint32_t at(std::size_t i, std::size_t j) const noexcept { return values[i * rows + j]; }

    std::uint32_t& at(std::size_t i, std::size_t j) noexcept { return values[i * rows + j]; }
};

struct RestartResult {
    std::vector<std::size_t> rows;
    std::int64_t score = std::numeric_limits<std::int64_t>::min();
    std::size_t accepted_moves = 0;
    std::size_t iterations_run = 0;
    std::size_t best_iteration = 0;
};

bool is_valid_probability(double value) noexcept { return value > 0.0 && value < 1.0; }

bool is_better_result(std::int64_t lhs_score,
                      const std::vector<std::size_t>& lhs_rows,
                      std::int64_t rhs_score,
                      const std::vector<std::size_t>& rhs_rows) {
    if (lhs_score != rhs_score)
        return lhs_score > rhs_score;
    return lhs_rows < rhs_rows;
}

std::vector<std::size_t> sorted_copy(const std::vector<std::size_t>& rows) {
    std::vector<std::size_t> out = rows;
    std::sort(out.begin(), out.end());
    return out;
}

void initialize_partition(std::size_t row_count,
                          std::size_t window_size,
                          XorShift64& rng,
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
                                    XorShift64& rng) {
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
    const double p0 = config.initial_acceptance_probability;
    const double temperature = -avg_negative_delta / std::log(p0);

    return std::max(temperature, config.min_temperature);
}

RestartResult run_restart(const PairwiseSynergyTable& table,
                          std::size_t window_size,
                          const SASelectorConfig& config,
                          XorShift64& rng) {
    std::vector<std::size_t> window;
    std::vector<std::size_t> pool;
    initialize_partition(table.rows, window_size, rng, window, pool);

    std::int64_t score = compute_window_score(window, table);

    RestartResult best;
    best.rows = sorted_copy(window);
    best.score = score;
    best.iterations_run = pool.empty() ? 0 : config.iterations;
    best.best_iteration = 0;
    best.accepted_moves = 0;

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

} // namespace

SASelectionResult SASelector::select(const BitMatrix& matrix, std::size_t window_size) const {
    if (_config.restarts == 0)
        throw std::invalid_argument("SASelector: restarts must be >= 1");

    if (!is_valid_probability(_config.initial_acceptance_probability)) {
        throw std::invalid_argument("SASelector: initial_acceptance_probability must be in (0, 1)");
    }

    if (_config.cooling_rate <= 0.0 || _config.cooling_rate > 1.0) {
        throw std::invalid_argument("SASelector: cooling_rate must be in (0, 1]");
    }

    if (_config.min_temperature <= 0.0) {
        throw std::invalid_argument("SASelector: min_temperature must be > 0");
    }

    if (window_size > matrix.rows()) {
        throw std::invalid_argument("SASelector: window_size exceeds matrix row count");
    }

    SASelectionResult out;
    out.seed = (_config.seed == 0 ? kDefaultSeed : _config.seed);

    if (window_size == 0)
        return out;

    PairwiseSynergyTable table;
    switch (_config.score_policy) {
    case WindowScorePolicyKind::PairwiseSynergy:
        table = build_pairwise_synergy_table(matrix);
        break;
    default:
        throw std::invalid_argument("SASelector: unsupported score policy");
    }

    switch (_config.cooling_policy) {
    case CoolingPolicyKind::AdaptiveGeometric:
        break;
    default:
        throw std::invalid_argument("SASelector: unsupported cooling policy");
    }

    XorShift64 rng(out.seed);
    bool has_best = false;

    for (std::size_t restart = 0; restart < _config.restarts; ++restart) {
        const RestartResult candidate = run_restart(table, window_size, _config, rng);

        if (!has_best || is_better_result(candidate.score, candidate.rows, out.score, out.rows)) {
            out.rows = candidate.rows;
            out.score = candidate.score;
            out.accepted_moves = candidate.accepted_moves;
            out.iterations_run = candidate.iterations_run;
            out.best_iteration = candidate.best_iteration;
            out.restart_index = restart;
            has_best = true;
        }
    }

    return out;
}

RowWindow SASelector::select_window(BitMatrix& matrix, std::size_t window_size) const {
    return matrix.row_window(select(matrix, window_size).rows);
}

RowWindow SASelector::select_window(const BitMatrix& matrix, std::size_t window_size) const {
    return matrix.row_window(select(matrix, window_size).rows);
}

} // namespace bmmpy
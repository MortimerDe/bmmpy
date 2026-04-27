#include "bmmpy/apply/greedy_applier.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>

namespace bmmpy {

ApplyResult GreedyApplier::apply(RowWindow& window, const std::vector<Candidate>& candidates) {

    const std::size_t n = window.size();
    if (n > 64)
        throw std::invalid_argument("row window size must be <= 64");

    std::array<std::uint64_t, 64> c{};
    for (std::size_t i = 0; i < n; ++i)
        c[i] = (1ULL << i);

    std::vector<std::uint64_t> current_weights(n);
    for (std::size_t i = 0; i < n; ++i)
        current_weights[i] = window.row_popcount(i);

    std::vector<const Candidate*> sorted_candidates;
    sorted_candidates.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
        sorted_candidates.push_back(&candidate);

    if (_stochastic) {
        for (std::size_t i = sorted_candidates.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(next_random() % i);
            std::swap(sorted_candidates[i - 1], sorted_candidates[j]);
        }
    }

    std::stable_sort(
        sorted_candidates.begin(),
        sorted_candidates.end(),
        [](const Candidate* lhs, const Candidate* rhs) { return lhs->weight < rhs->weight; });

    ApplyResult result;

    for (const Candidate* candidate : sorted_candidates) {
        const std::uint64_t m = candidate->mask_u64();

        std::uint64_t a = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if ((detail::parity64(m & c[i]) & 1u) != 0)
                a |= (1ULL << i);
        }

        if (a == 0)
            continue;

        std::array<std::size_t, 64> best_rows{};
        std::size_t best_rows_count = 0;
        std::uint64_t best_gain = 0;

        for (std::size_t r = 0; r < n; ++r) {
            if (((a >> r) & 1ULL) == 0)
                continue;

            const std::uint64_t old_weight = current_weights[r];
            const std::uint64_t candidate_weight = static_cast<std::uint64_t>(candidate->weight);

            if (old_weight <= candidate_weight)
                continue;

            const std::uint64_t gain = old_weight - candidate_weight;
            if (gain < _min_gain)
                continue;

            if (gain > best_gain) {
                best_gain = gain;
                best_rows[0] = r;
                best_rows_count = 1;
            } else if (gain == best_gain) {
                best_rows[best_rows_count++] = r;
            }
        }

        if (best_rows_count == 0)
            continue;

        const std::size_t chosen =
            _stochastic ? best_rows[static_cast<std::size_t>(next_random() % best_rows_count)]
                        : best_rows[0];

        for (std::size_t i = 0; i < n; ++i) {
            if (i != chosen && ((a >> i) & 1ULL) != 0)
                window.row_xor(chosen, i);
        }

        current_weights[chosen] = static_cast<std::uint64_t>(candidate->weight);

        for (std::size_t j = 0; j < n; ++j) {
            if (j != chosen && ((a >> j) & 1ULL) != 0)
                c[j] ^= c[chosen];
        }

        ++result.applied_count;
        result.weight_improvement += best_gain;
    }

    return result;
}

} // namespace bmmpy
#include "bmmpy/ga/genetic_algorithm.hpp"

#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <print>
#include <random>
#include <utility>
#include <vector>

namespace bmmpy::ga {

Individual GeneticAlgorithm::tournament_selection() {
    std::uniform_int_distribution<std::size_t> distribution(0, _population.size() - 1);

    std::size_t best_idx = distribution(_rng);
    for (std::size_t i = 1; i < _config.tournament_size; ++i) {
        const std::size_t candidate_idx = distribution(_rng);
        if (_fitnesses[candidate_idx] < _fitnesses[best_idx])
            best_idx = candidate_idx;
    }

    return _population[best_idx];
}

Individual GeneticAlgorithm::crossover(const Individual& lhs, const Individual& rhs) {
    const auto crossover_started = internal::steady_clock::now();
    // std::println("[ga:crossover:start] lhs={} rhs={} elapsed_ms=0", lhs.size(), rhs.size());

    std::vector<const Candidate*> pool;
    pool.reserve(lhs.size() + rhs.size());

    for (const Candidate& candidate : lhs)
        pool.push_back(&candidate);
    for (const Candidate& candidate : rhs)
        pool.push_back(&candidate);

    std::stable_sort(pool.begin(), pool.end(), [](const Candidate* left, const Candidate* right) {
        if (left->weight != right->weight)
            return left->weight < right->weight;
        return left->mask_popcount() < right->mask_popcount();
    });

    const std::size_t bit_width = lhs.size();
    ::bmmpy::detail::PivotBasis pivot(bit_width);
    Individual child;
    child.reserve(bit_width);

    for (const Candidate* candidate : pool) {
        if (pivot.try_insert(candidate->mask))
            child.push_back(*candidate);
        if (child.size() == bit_width)
            break;
    }

    for (std::size_t i = 0; i < bit_width && child.size() < bit_width; ++i) {
        auto unit_mask = ::bmmpy::detail::make_unit_mask_words(bit_width, i);
        if (pivot.try_insert(unit_mask))
            child.push_back(Candidate(std::move(unit_mask), 1));
    }

    // std::println("[ga:crossover:done] child_size={} rank={} elapsed_ms={}",
    //              child.size(),
    //              pivot.rank(),
    //              internal::elapsed_ms(crossover_started));
    return child;
}


void GeneticAlgorithm::mutate(Individual& ind) {
    const auto mutate_started = internal::steady_clock::now();
    const std::size_t candidate_count = ind.size();

    if (candidate_count < 2) {
        // std::println("[ga:mutate:start] n=0 rows={} elapsed_ms=0", candidate_count);
        // std::println("[ga:mutate:done] n=0 elapsed_ms={}", internal::elapsed_ms(mutate_started));
        return;
    }

    const std::size_t mutation_count =
        std::max<std::size_t>(1, static_cast<std::size_t>(candidate_count * _config.mutation_rate));
    std::uniform_int_distribution<std::size_t> distribution(0, candidate_count - 1);

    ::bmmpy::BitMatrix scratch_storage;
    std::uint64_t* scratch_words = nullptr;
    if (_M != 0) {
        scratch_storage = ::bmmpy::BitMatrix(1, _M);
        scratch_words = scratch_storage.row_words(0);
    }

    // std::println("[ga:mutate:start] n={} rows={} elapsed_ms=0", mutation_count, candidate_count);

    for (std::size_t k = 0; k < mutation_count; ++k) {
        const std::size_t i = distribution(_rng);
        const std::size_t j = distribution(_rng);
        if (i == j)
            continue;

        // std::println("[ga:mutate:iter:start] iter={} i={} j={} elapsed_ms={}",
                     // k,
                     // i,
                     // j,
                     // internal::elapsed_ms(mutate_started));

        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];

        ind[i].weight = internal::eval_cand_weight(*_window, _N, _M, ind[i], scratch_words);

        // std::println(
            // "[ga:mutate:iter:done] iter={} elapsed_ms={}", k, internal::elapsed_ms(mutate_started));
    }

    // std::println("[ga:mutate:done] n={} elapsed_ms={}",
    //              mutation_count,
    //              internal::elapsed_ms(mutate_started));
}

void GeneticAlgorithm::local_improvement(Individual& ind) {
    const std::size_t candidate_count = ind.size();
    if (candidate_count < 2 || _M == 0)
        return;

    ::bmmpy::BitMatrix materialized(candidate_count, _M);

    for (std::size_t i = 0; i < candidate_count; ++i) {
        internal::mat_cand(*_window, _N, ind[i], materialized.row_words(i));
        ind[i].weight = static_cast<std::uint32_t>(materialized.row_popcount(i));
    }

    const auto& ops = ::bmmpy::detail::bit_ops();
    const std::size_t word_count = materialized.words_per_row();
    constexpr std::size_t k_local_improvement_iters = 10;

    for (std::size_t iter = 0; iter < k_local_improvement_iters; ++iter) {
        bool improved = false;

        for (std::size_t i = 0; i < candidate_count; ++i) {
            const std::uint64_t wi = ind[i].weight;
            if (wi == 0)
                continue;

            const std::uint64_t* const row_i = materialized.row_words(i);

            std::size_t best_j = candidate_count;
            std::uint64_t best_weight = wi;

            for (std::size_t j = 0; j < candidate_count; ++j) {
                if (i == j)
                    continue;

                const std::uint64_t wj = ind[j].weight;
                if (wj >= 2ull * wi)
                    continue;

                const std::uint64_t overlap =
                    ops.row_and_popcount(row_i, materialized.row_words(j), word_count);
                const std::uint64_t candidate_weight = wi + wj - 2ull * overlap;

                if (candidate_weight < best_weight) {
                    best_weight = candidate_weight;
                    best_j = j;
                }
            }

            if (best_j == candidate_count)
                continue;

            for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
                ind[i].mask[w] ^= ind[best_j].mask[w];

            materialized.row_xor(i, best_j);
            ind[i].weight = static_cast<std::uint32_t>(best_weight);
            improved = true;
        }

        if (!improved)
            break;
    }
}

} // namespace bmmpy::ga

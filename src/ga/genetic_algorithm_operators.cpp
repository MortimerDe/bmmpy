#include "bmmpy/ga/genetic_algorithm.hpp"

#include "bmmpy/ga/genetic_algorithm_internal.hpp"

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
    std::println("[ga:crossover:start] lhs={} rhs={} elapsed_ms=0", lhs.size(), rhs.size());

    std::vector<const Candidate*> pool;
    pool.reserve(lhs.size() + rhs.size());

    for (const Candidate& candidate : lhs)
        pool.push_back(&candidate);
    for (const Candidate& candidate : rhs)
        pool.push_back(&candidate);

    std::stable_sort(pool.begin(), pool.end(), [](const Candidate* left, const Candidate* right) {
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

    std::println("[ga:crossover:done] child_size={} rank={} elapsed_ms={}",
                 child.size(),
                 pivot.rank(),
                 internal::elapsed_ms(crossover_started));
    return child;
}

void GeneticAlgorithm::mutate(Individual& ind) {
    const auto mutate_started = internal::steady_clock::now();
    const std::size_t mutation_count = std::max<std::size_t>(1, _N * _config.mutation_rate);
    std::uniform_int_distribution<std::size_t> distribution(0, _N - 1);

    std::println("[ga:mutate:start] n={} rows={} elapsed_ms=0", mutation_count, _N);

    for (std::size_t k = 0; k < mutation_count; ++k) {
        const std::size_t i = distribution(_rng);
        const std::size_t j = distribution(_rng);
        if (i == j)
            continue;

        std::println("[ga:mutate:iter:start] iter={} i={} j={} elapsed_ms={}",
                     k,
                     i,
                     j,
                     internal::elapsed_ms(mutate_started));

        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];

        recalc_all_weights(ind);

        std::println(
            "[ga:mutate:iter:done] iter={} elapsed_ms={}", k, internal::elapsed_ms(mutate_started));
    }

    std::println("[ga:mutate:done] n={} elapsed_ms={}",
                 mutation_count,
                 internal::elapsed_ms(mutate_started));
}

// todo: слишком медленный, надо доработать
void GeneticAlgorithm::local_improvement(Individual& ind) {
    const auto improve_started = internal::steady_clock::now();
    std::println("[ga:local:start] rows={} elapsed_ms=0", _N);

    for (std::size_t iter = 0; iter < 10; ++iter) {
        bool improved = false;
        std::println("[ga:local:iter:start] iter={} elapsed_ms={}",
                     iter,
                     internal::elapsed_ms(improve_started));

        for (std::size_t i = 0; i < _N; ++i) {
            for (std::size_t j = 0; j < _N; ++j) {
                if (i == j)
                    continue;

                Candidate::mask_type old_mask = ind[i].mask;
                const std::uint32_t old_weight = ind[i].weight;

                for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
                    ind[i].mask[w] ^= ind[j].mask[w];

                const std::uint32_t new_weight = internal::eval_cand_weight(*_window, _N, _M, ind[i]);

                if (new_weight < old_weight) {
                    ind[i].weight = new_weight;
                    std::println("[ga:local:improved] iter={} i={} j={} old_w={} new_w={} "
                                 "elapsed_ms={}",
                                 iter,
                                 i,
                                 j,
                                 old_weight,
                                 new_weight,
                                 internal::elapsed_ms(improve_started));
                    improved = true;
                    break;
                }

                ind[i].mask = std::move(old_mask);
                ind[i].weight = old_weight;
            }

            if (improved)
                break;
        }

        if (!improved)
            break;
    }

    std::println("[ga:local:done] elapsed_ms={}", internal::elapsed_ms(improve_started));
}

} // namespace bmmpy::ga

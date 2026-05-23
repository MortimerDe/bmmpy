#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"
#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
// #include <print>
#include <utility>
#include <vector>

namespace bmmpy::ga {

void GeneticAlgorithm::catastrophe() {
    std::size_t survivors_count = _population.size() * _config.catastrophe_survival_rate;

    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    for (std::size_t i = 0; i < _population.size(); ++i)
        ranked.emplace_back(_fitnesses[i], i);
    std::sort(ranked.begin(), ranked.end());

    std::vector<Individual> survivors;
    survivors.reserve(_population.size());
    for (std::size_t i = 0; i < survivors_count && i < ranked.size(); ++i)
        survivors.push_back(_population[ranked[i].second]);

    while (survivors.size() < _config.population_size)
        survivors.push_back(make_random());

    _population = std::move(survivors);

    for (std::size_t i = 0; i < _population.size(); ++i)
        _fitnesses[i] = evaluate_individual(_population[i]);

    _catastrophe = 0;
}

std::vector<Individual> GeneticAlgorithm::tournament_selection() {
    std::vector<Individual> parents;
    parents.reserve(_config.num_parents);

    for (std::size_t p = 0; p < _config.num_parents; ++p) {
        std::size_t best_idx = _rng.next_index(_population.size());

        for (std::size_t t = 1; t < _config.tournament_size; ++t) {
            const std::size_t candidate_idx = _rng.next_index(_population.size());
            if (_fitnesses[candidate_idx] < _fitnesses[best_idx])
                best_idx = candidate_idx;
        }

        parents.push_back(_population[best_idx]);
    }

    return parents;
}

std::vector<Individual> GeneticAlgorithm::crossover(const std::vector<Individual>& parents) {

    std::vector<const Candidate*> pool;
    for (const Individual& individual : parents) {
        pool.reserve(pool.size() + individual.size());
        for (const Candidate& candidate : individual) {
            pool.push_back(&candidate);
        }
    }

    std::stable_sort(pool.begin(), pool.end(), [](const Candidate* left, const Candidate* right) {
        if (left->weight != right->weight)
            return left->weight < right->weight;
        return left->mask_popcount() < right->mask_popcount();
    });

    const std::size_t bit_width = parents[0].size();
    std::vector<Individual> offsprings;
    offsprings.reserve(_config.num_offspring);

    for (std::size_t p = 0; p < _config.num_offspring; ++p) {
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

        offsprings.push_back(child);
    }
    return offsprings;
}

void GeneticAlgorithm::mutate(Individual& ind) {
    const auto mutate_started = internal::steady_clock::now();
    const std::size_t candidate_count = ind.size();

    if (candidate_count < 2) {
        return;
    }

    const std::size_t mutation_count =
        std::max<std::size_t>(1, static_cast<std::size_t>(candidate_count * _config.mutation_rate));

    ::bmmpy::BitMatrix scratch_storage;
    std::uint64_t* scratch_words = nullptr;
    if (_M != 0) {
        scratch_storage = ::bmmpy::BitMatrix(1, _M);
        scratch_words = scratch_storage.row_words(0);
    }

    for (std::size_t k = 0; k < mutation_count; ++k) {
        const std::size_t i = _rng.next_index(candidate_count);
        const std::size_t j = _rng.next_index(candidate_count);
        if (i == j)
            continue;

        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];

        ind[i].weight = internal::eval_cand_weight(*_window, _N, _M, ind[i], scratch_words);
    }
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

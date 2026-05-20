#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include <algorithm>
#include <cstddef>
// #include <print>

namespace bmmpy::ga {

Individual GeneticAlgorithm::make_identity() const {
    Individual individual;
    individual.reserve(_N);

    for (std::size_t i = 0; i < _N; ++i)
        individual.push_back(Candidate::make_unit(_N, i, _window->row_popcount(i)));

    return individual;
}

Individual GeneticAlgorithm::make_random() {
    const auto random_started = internal::steady_clock::now();
    Individual individual = make_identity();

    for (std::size_t k = 0; k < _N * 3; ++k) {
        const std::size_t i = _rng.next_index(_N);
        const std::size_t j = _rng.next_index(_N);
        if (i == j)
            continue;

        for (std::size_t w = 0; w < individual[i].mask.size(); ++w)
            individual[i].mask[w] ^= individual[j].mask[w];
    }

    recalc_all_weights(individual);
    return individual;
}

Individual GeneticAlgorithm::make_heuristic() {
    Individual individual = make_identity();
    if (_N < 2)
        return individual;

    std::vector<std::size_t> sorted_indices(_N);
    for (std::size_t i = 0; i < _N; ++i)
        sorted_indices[i] = i;

    std::stable_sort(
        sorted_indices.begin(), sorted_indices.end(), [&](std::size_t lhs, std::size_t rhs) {
            return individual[lhs].weight > individual[rhs].weight;
        });

    const std::size_t probe_count = std::min<std::size_t>(5, _N);

    ::bmmpy::BitMatrix scratch_storage(1, _M);
    std::uint64_t* const scratch_words = scratch_storage.row_words(0);

    for (std::size_t pos = 0; pos < probe_count; ++pos) {
        const std::size_t i = sorted_indices[pos];
        std::uint32_t current_weight = individual[i].weight;

        for (std::size_t j = 0; j < _N; ++j) {
            if (i == j)
                continue;

            for (std::size_t w = 0; w < individual[i].mask.size(); ++w)
                individual[i].mask[w] ^= individual[j].mask[w];

            const std::uint32_t candidate_weight =
                internal::eval_cand_weight(*_window, _N, _M, individual[i], scratch_words);

            if (candidate_weight < current_weight) {
                individual[i].weight = candidate_weight;
                current_weight = candidate_weight;
                break;
            }

            for (std::size_t w = 0; w < individual[i].mask.size(); ++w)
                individual[i].mask[w] ^= individual[j].mask[w];
        }
    }

    return individual;
}

} // namespace bmmpy::ga

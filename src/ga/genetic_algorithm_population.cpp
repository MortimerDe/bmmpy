#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include <cstddef>
#include <print>
#include <random>

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
    std::println("[ga:random:start] rows={} cols={} elapsed_ms=0", _N, _M);

    Individual individual = make_identity();
    std::uniform_int_distribution<std::size_t> distribution(0, _N - 1);

    for (std::size_t k = 0; k < _N * 3; ++k) {
        const std::size_t i = distribution(_rng);
        const std::size_t j = distribution(_rng);
        if (i == j)
            continue;

        std::println("[ga:random:xor] iter={} i={} j={} elapsed_ms={}",
                     k,
                     i,
                     j,
                     internal::elapsed_ms(random_started));

        for (std::size_t w = 0; w < individual[i].mask.size(); ++w)
            individual[i].mask[w] ^= individual[j].mask[w];
    }

    std::println("[ga:random] before_recalc elapsed_ms={}", internal::elapsed_ms(random_started));
    recalc_all_weights(individual);

    std::println("[ga:random:done] score={} elapsed_ms={}",
                 evaluate_individual(individual),
                 internal::elapsed_ms(random_started));
    return individual;
}

} // namespace bmmpy::ga

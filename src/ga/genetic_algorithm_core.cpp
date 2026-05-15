#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include "bmmpy/core/bit_matrix.hpp"

#include <algorithm>
#include <cstddef>

namespace bmmpy::ga {

std::size_t GeneticAlgorithm::evaluate_individual(const Individual& ind) const {
    std::size_t total = 0;
    for (const Candidate& candidate : ind)
        total += candidate.weight;
    return total;
}

void GeneticAlgorithm::recalc_all_weights(Individual& ind) {
    const auto recalc_started = internal::steady_clock::now();
    // std::println("[ga:recalc:start] candidates={} rows={} cols={}", ind.size(), _N, _M);

    ::bmmpy::BitMatrix scratch_storage;
    std::uint64_t* scratch_words = nullptr;

    if (_M != 0) {
        scratch_storage = ::bmmpy::BitMatrix(1, _M);
        scratch_words = scratch_storage.row_words(0);
    }

    for (Candidate& candidate : ind) {
        candidate.weight = internal::eval_cand_weight(*_window, _N, _M, candidate, scratch_words);

        // const std::size_t candidate_idx = static_cast<std::size_t>(&candidate - ind.data());
        // std::println("[ga:recalc:candidate] idx={} weight={} elapsed_ms={}",
        //              candidate_idx,
        //              candidate.weight,
        //              internal::elapsed_ms(recalc_started));
    }

    // std::println("[ga:recalc:done] candidates={} elapsed_ms={}",
    //              ind.size(),
    //              internal::elapsed_ms(recalc_started));
}

void GeneticAlgorithm::adapt_mutation_rate() {
    if (_no_improvement > 10) {
        _config.mutation_rate = std::min(0.8, _config.mutation_rate * 1.5);
    } else if (_no_improvement > 5) {
        _config.mutation_rate = std::min(0.6, _config.mutation_rate * 1.3);
    } else {
        _config.mutation_rate = std::max(0.3, _config.mutation_rate * 0.95);
    }
}

} // namespace bmmpy::ga

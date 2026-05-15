#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/algorithm.hpp"
#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace bmmpy::ga {

struct GeneticAlgorithmConfig {
    std::size_t population_size = 300;
    std::size_t elite_count = 3;
    std::size_t tournament_size = 3;
    double mutation_rate = 0.3;
    StopCriteria stop;
    std::uint64_t seed = 0;
};

class GeneticAlgorithm final : public Algorithm {
public:
    explicit GeneticAlgorithm(GeneticAlgorithmConfig config = {});

    void initialize(const ::bmmpy::RowWindow& window) override;
    void step() override;

    bool done() const noexcept override;
    std::size_t generation() const noexcept override;

    Individual best_individual() const override;
    RunStats stats() const override;

    std::vector<Individual> export_migrants(std::size_t max_count) override;
    void import_migrants(std::vector<Individual> migrants) override;

    std::size_t best_score() const override;

    Individual optimize(const ::bmmpy::RowWindow& window);

    const char* name() const noexcept override { return "ga"; }

private:
    // Fitness
    std::size_t evaluate_individual(const Individual& ind) const;
    void recalc_all_weights(Individual& ind);
    void adapt_mutation_rate();

    // Operators
    Individual tournament_selection();
    Individual crossover(const Individual& a, const Individual& b);
    void mutate(Individual& ind);
    void local_improvement(Individual& ind);

    // Initialization
    Individual make_identity() const;
    Individual make_random();

    GeneticAlgorithmConfig _config;
    RunStats _stats{};
    Individual _best_individual;
    std::size_t _best_score = 0;
    bool _initialized = false;
    bool _done = false;

    const RowWindow* _window = nullptr;
    std::size_t _N = 0;
    std::size_t _M = 0;

    std::vector<Individual> _population;
    std::vector<std::size_t> _fitnesses;

    std::size_t _no_improvement = 0;
    std::mt19937 _rng;
};

} // namespace bmmpy::ga
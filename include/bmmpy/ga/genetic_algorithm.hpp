#pragma once

#include "bmmpy/core/detail/xorshift64.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/algorithm.hpp"
#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace bmmpy::ga {

enum class LocalImprStrategy { DISABLED, GREEDY, FWHT, CUDA_FWHT, BRUTE, CUDA_BRUTE };

struct GeneticAlgorithmConfig {
    std::size_t population_size = 300;
    std::size_t elite_count = 3;
    std::size_t num_parents = 2;
    std::size_t num_offspring = 1;
    std::size_t tournament_size = 3;
    bool enable_catastrophe = false;
    std::size_t catastrophe_threshold = 30;
    double catastrophe_survival_rate = 0.2;
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
    void catastrophe();
    std::vector<Individual> tournament_selection();
    std::vector<Individual> crossover(const std::vector<Individual>& parents);
    void mutate(Individual& ind);
    void local_improvement(Individual& ind);

    // Initialization
    Individual make_identity() const;
    Individual make_random();
    Individual make_heuristic();

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
    detail::XorShift64 _rng;
    std::size_t _catastrophe = 0;
    std::mt19937 _rng;
};

} // namespace bmmpy::ga

#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/island.hpp"
#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace bmmpy::ga {
struct GeneticAlgorithmConfig {
    std::size_t population_size = 128;
    std::size_t elite_count = 8;
    std::size_t tournament_size = 4;

    double crossover_rate = 0.9;
    double mutation_rate = 0.05;

    StopCriteria stop;
    std::uint64_t seed = 0;
};

class GeneticAlgorithm final : public Island {
public:
    explicit GeneticAlgorithm(GeneticAlgorithmConfig config = {});

    void initialize(const ::bmmpy::RowWindow& window) override;
    void step_generation() override;

    bool done() const noexcept override;
    std::size_t generation() const noexcept override;

    Individual best_individual() const override;
    RunStats stats() const override;

    std::vector<Individual> export_migrants(std::size_t max_count) override;
    void import_migrants(std::vector<Individual> migrants) override;

    std::unique_ptr<Island> clone() const override;

    Individual optimize(const ::bmmpy::RowWindow& window);

    const char* name() const noexcept override { return "ga"; }

private:
    GeneticAlgorithmConfig config_;
};
} // namespace bmmpy::ga
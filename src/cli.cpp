#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/island_model.hpp"
#include "bmmpy/ga/migration/bounded_channel.hpp"
#include "bmmpy/stub.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <print>
#include <string>
#include <vector>

namespace {

bmmpy::BitMatrix load_matrix_from_file(const std::string& filename) {
    try {
        auto matrix = bmmpy::BitMatrix::load_text(filename);
        std::println("Loaded matrix from '{}': {}x{}, weight={}",
                     filename,
                     matrix.rows(),
                     matrix.cols(),
                     matrix.weight());
        return matrix;
    } catch (const std::exception& ex) {
        std::println("Error loading matrix from '{}': {}", filename, ex.what());
        throw;
    }
}

bmmpy::BitMatrix make_demo_matrix() {
    bmmpy::BitMatrix matrix(4, 8);
    matrix.set(0, 0, true);
    matrix.set(0, 1, true);
    matrix.set(0, 4, true);
    matrix.set(1, 1, true);
    matrix.set(1, 2, true);
    matrix.set(1, 5, true);
    matrix.set(2, 2, true);
    matrix.set(2, 5, true);
    matrix.set(2, 6, true);
    matrix.set(3, 0, true);
    matrix.set(3, 3, true);
    matrix.set(3, 7, true);
    return matrix;
}

void print_matrix(const std::string& title, const bmmpy::BitMatrix& matrix) {
    std::println("{} ({}x{}, weight={}):", title, matrix.rows(), matrix.cols(), matrix.weight());
    for (std::size_t r = 0; r < matrix.rows(); ++r) {
        std::print("  row {}: ", r);
        for (std::size_t c = 0; c < matrix.cols(); ++c) {
            std::print("{}", matrix.get(r, c) ? '1' : '0');
        }
        std::println(" (w={})", matrix.row_popcount(r));
    }
}

void print_individual(const bmmpy::ga::Individual& ind, const bmmpy::RowWindow& window) {
    std::println("Individual ({} candidates):", ind.size());
    for (std::size_t i = 0; i < ind.size(); ++i) {
        std::print("  T[{}] = [", i);
        for (std::size_t j = 0; j < window.size(); ++j) {
            std::print("{}", ind[i].has_row(j) ? '1' : '0');
        }
        std::println("] weight={}", ind[i].weight);
    }
}

bmmpy::BitMatrix apply_individual(const bmmpy::BitMatrix& original,
                                  const bmmpy::RowWindow& window,
                                  const bmmpy::ga::Individual& ind) {
    const std::size_t N = window.size();
    const std::size_t M = original.cols();

    bmmpy::BitMatrix result(N, M);

    for (std::size_t out = 0; out < N; ++out) {
        std::vector<bool> row(M, false);
        for (std::size_t src = 0; src < N; ++src) {
            if (ind[out].has_row(src)) {
                for (std::size_t c = 0; c < M; ++c) {
                    row[c] = row[c] != window.get(src, c);
                }
            }
        }
        for (std::size_t c = 0; c < M; ++c) {
            result.set(out, c, row[c]);
        }
    }

    return result;
}

bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

} // namespace

bmmpy::ga::GeneticAlgorithmConfig make_ga_config_for_island(std::size_t island_id,
                                                            std::uint64_t base_seed) {
    bmmpy::ga::GeneticAlgorithmConfig cfg;

    cfg.stop.max_generations = 50;
    cfg.stop.max_stale_generations = 50;

    switch (island_id % 4) {
    case 0:
        cfg.population_size = 500;
        cfg.num_parents = 2;
        cfg.num_offspring = 1;
        cfg.elite_count = 3;
        cfg.tournament_size = 2;
        cfg.mutation_rate = 0.20;
        cfg.seed = base_seed + 1;
        cfg.catastrophe_threshold = 30;
        cfg.catastrophe_survival_rate = 0.8;
        break;

    case 1:
        cfg.population_size = 500;
        cfg.num_parents = 4;
        cfg.num_offspring = 2;
        cfg.elite_count = 3;
        cfg.tournament_size = 2;
        cfg.mutation_rate = 0.30;
        cfg.seed = base_seed + 2;
        cfg.catastrophe_threshold = 30;
        cfg.catastrophe_survival_rate = 0.5;
        break;

    case 2:
        cfg.population_size = 500;
        cfg.num_parents = 2;
        cfg.num_offspring = 1;
        cfg.elite_count = 4;
        cfg.tournament_size = 3;
        cfg.mutation_rate = 0.45;
        cfg.seed = base_seed + 3;
        cfg.catastrophe_threshold = 30;
        cfg.catastrophe_survival_rate = 0.5;
        break;

    default:
        cfg.population_size = 500;
        cfg.num_parents = 10;
        cfg.num_offspring = 3;
        cfg.elite_count = 2;
        cfg.tournament_size = 10;
        cfg.mutation_rate = 1.0;
        cfg.seed = base_seed + 4;
        cfg.catastrophe_threshold = 30;
        cfg.catastrophe_survival_rate = 0.5;
        break;
    }

    return cfg;
}

bmmpy::ga::IslandModelConfig make_island_model_config() {
    bmmpy::ga::IslandModelConfig model_cfg;

    {
        bmmpy::ga::IslandSpec spec;
        spec.island_id = 0;
        spec.migration.interval_generations = 12;
        spec.migration.export_count = 2;
        spec.migration.import_count = 2;
        model_cfg.islands.push_back(spec);
    }

    {
        bmmpy::ga::IslandSpec spec;
        spec.island_id = 1;
        spec.migration.interval_generations = 8;
        spec.migration.export_count = 2;
        spec.migration.import_count = 2;
        model_cfg.islands.push_back(spec);
    }

    {
        bmmpy::ga::IslandSpec spec;
        spec.island_id = 2;
        spec.migration.interval_generations = 16;
        spec.migration.export_count = 3;
        spec.migration.import_count = 1;
        model_cfg.islands.push_back(spec);
    }

    {
        bmmpy::ga::IslandSpec spec;
        spec.island_id = 3;
        spec.migration.interval_generations = 10;
        spec.migration.export_count = 1;
        spec.migration.import_count = 3;
        model_cfg.islands.push_back(spec);
    }

    return model_cfg;
}

void print_island_model_snapshot(const bmmpy::ga::IslandModelSnapshot& snapshot) {
    std::uint64_t total_evaluations = 0;

    std::println("Island model summary:");
    std::println(
        "  running={} stop_requested={} island_count={} total_generations={} best_score={}",
        snapshot.running,
        snapshot.stop_requested,
        snapshot.island_count,
        snapshot.total_generations,
        snapshot.best_score);

    for (const auto& island : snapshot.islands) {
        total_evaluations += island.stats.evaluations;

        std::println("  island={} best={} gen={} evals={} stale={} running={} finished={}",
                     island.island_id,
                     island.best_score,
                     island.stats.generations,
                     island.stats.evaluations,
                     island.stats.stale_generations,
                     island.running,
                     island.finished);
    }

    std::println("  total_evaluations={}", total_evaluations);
}

int main(int argc, char* argv[]) {
    std::println("=== BM-MPY Genetic Algorithm ===\n");
    std::println("Version: {}", bmmpy::get_version());

    bmmpy::BitMatrix matrix;

    std::string filename = "___matrix.txt";
    if (argc > 1) {
        filename = argv[1];
    }

    if (file_exists(filename)) {
        std::println("Loading matrix from '{}'...", filename);
        try {
            matrix = load_matrix_from_file(filename);
        } catch (const std::exception& ex) {
            std::println("Failed to load '{}', using demo matrix", filename);
            matrix = make_demo_matrix();
        }
    } else {
        std::println("File '{}' not found, using demo matrix", filename);
        matrix = make_demo_matrix();
    }

    std::vector<std::size_t> row_indices(matrix.rows());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    bmmpy::RowWindow window = matrix.row_window(row_indices);
    std::println("Window size: {}\n", window.size());

    constexpr bool k_use_island_model = true;
    const std::uint64_t base_seed = 42;

    bmmpy::ga::Individual best_individual;
    std::size_t best_score = 0;

    std::println("Starting optimization...\n");

    if (k_use_island_model) {
        const bmmpy::ga::IslandModelConfig model_cfg = make_island_model_config();

        bmmpy::ga::AlgorithmFactory factory =
            [base_seed](
                const bmmpy::ga::IslandSpec& spec) -> std::unique_ptr<bmmpy::ga::Algorithm> {
            return std::make_unique<bmmpy::ga::GeneticAlgorithm>(
                make_ga_config_for_island(spec.island_id, base_seed));
        };

        auto channel = std::make_unique<bmmpy::ga::migration::BoundedChannel>(
            256, bmmpy::ga::migration::OverflowPolicy::DropOldest);

        bmmpy::ga::IslandModel island_model(model_cfg, std::move(factory), std::move(channel));

        best_individual = island_model.run_to_completion(window);

        const bmmpy::ga::IslandModelSnapshot snapshot = island_model.snapshot();
        best_score = snapshot.best_score;

        std::println("\n=== Results (Island Model) ===");
        print_island_model_snapshot(snapshot);
    } else {
        bmmpy::ga::GeneticAlgorithmConfig config;
        config.population_size = 32;
        config.mutation_rate = 0.3;
        config.elite_count = 3;
        config.tournament_size = 2;
        config.seed = base_seed;
        config.stop.max_generations = 400;
        config.stop.max_stale_generations = 120;

        bmmpy::ga::GeneticAlgorithm ga(config);

        best_individual = ga.optimize(window);
        best_score = ga.best_score();

        const auto stats = ga.stats();

        std::println("\n=== Results (Single GA) ===");
        std::println("Generations: {}", stats.generations);
        std::println("Evaluations: {}", stats.evaluations);
        std::println("Best score: {}", best_score);
    }

    auto result = apply_individual(matrix, window, best_individual);
    const std::uint64_t result_weight = result.weight();
    const std::int64_t delta =
        static_cast<std::int64_t>(matrix.weight()) - static_cast<std::int64_t>(result_weight);

    std::println("Applied result weight: {}", result_weight);
    if (result_weight != best_score) {
        std::println("WARNING: cached best score != applied result weight ({} != {})",
                     best_score,
                     result_weight);
    }

    if (delta >= 0) {
        std::println("Improvement: {} -> {} (-{})\n", matrix.weight(), result_weight, delta);
    } else {
        std::println("Worsening: {} -> {} (+{})\n", matrix.weight(), result_weight, -delta);
    }

    std::string output_filename = "optimized_matrix.txt";
    try {
        result.save_text(output_filename);
        std::println("\nResult saved to '{}'", output_filename);
    } catch (const std::exception& ex) {
        std::println("\nCould not save result: {}", ex.what());
    }

    return 0;
}

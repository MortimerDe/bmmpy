#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/island_model.hpp"
#include "bmmpy/ga/migration/bounded_channel.hpp"
#include "bmmpy/stub.hpp"

#include <iostream>
#include <memory>
// #include <print>
#include <vector>

namespace {

bmmpy::BitMatrix make_demo_matrix() {
    bmmpy::BitMatrix matrix(4, 8);

    matrix.set(0, 0, true);
    matrix.set(0, 1, true);
    matrix.set(0, 4, true);

    matrix.set(1, 1, true);
    matrix.set(1, 2, true);
    matrix.set(1, 5, true);

    matrix.set(2, 2, true);
    matrix.set(2, 3, true);
    matrix.set(2, 6, true);

    matrix.set(3, 0, true);
    matrix.set(3, 3, true);
    matrix.set(3, 7, true);

    return matrix;
}

void print_snapshot(const bmmpy::ga::IslandModelSnapshot& snapshot) {
    std::cout << "model.running=" << snapshot.running << "\n";
    std::cout << "model.stop_requested=" << snapshot.stop_requested << "\n";
    std::cout << "model.island_count=" << snapshot.island_count << "\n";
    std::cout << "model.total_generations=" << snapshot.total_generations << "\n";
    std::cout << "model.best_score=" << snapshot.best_score << "\n";

    for (const auto& island : snapshot.islands) {
        std::cout << "  island[" << island.island_id << "]"
                  << " running=" << island.running << " finished=" << island.finished
                  << " generations=" << island.stats.generations
                  << " best_score=" << island.best_score << "\n";
    }
}

} // namespace

int main() {
    std::cout << "Version: " << bmmpy::get_version() << "\n";

    bmmpy::BitMatrix matrix = make_demo_matrix();
    bmmpy::RowWindow window = matrix.row_window(std::vector<std::size_t>{0, 1, 2, 3});

    std::vector<bmmpy::ga::GeneticAlgorithmConfig> configs(3);
    configs[0].seed = 11;
    configs[0].stop.max_generations = 8;

    configs[1].seed = 22;
    configs[1].stop.max_generations = 12;

    configs[2].seed = 33;
    configs[2].stop.max_generations = 16;

    bmmpy::ga::IslandModelConfig model_config{{
        {0,
         {.interval_generations = 2,
          .export_count = 1,
          .import_count = 1,
          .shared_pool_capacity = 16}},
        {1,
         {.interval_generations = 3,
          .export_count = 1,
          .import_count = 1,
          .shared_pool_capacity = 16}},
        {2,
         {.interval_generations = 4,
          .export_count = 1,
          .import_count = 1,
          .shared_pool_capacity = 16}},
    }};

    bmmpy::ga::AlgorithmFactory factory =
        [configs](const bmmpy::ga::IslandSpec& spec) -> std::unique_ptr<bmmpy::ga::Algorithm> {
        return std::make_unique<bmmpy::ga::GeneticAlgorithm>(configs.at(spec.island_id));
    };

    auto channel = std::make_unique<bmmpy::ga::migration::BoundedChannel>(16);

    bmmpy::ga::IslandModel model(std::move(model_config), std::move(factory), std::move(channel));

    try {
        const auto best = model.run_to_completion(window);
        const auto snapshot = model.snapshot();

        print_snapshot(snapshot);
        std::cout << "best_individual.size=" << best.size() << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "ga runtime error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
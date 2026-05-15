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

// Загрузка матрицы из текстового файла
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

// Демо-матрица для тестирования
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

// Печать матрицы
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

// Печать Individual
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

// Применение Individual к матрице через RowWindow
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

// Проверка существования файла
bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

} // namespace

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

    // print_matrix("Original matrix", matrix);

    std::vector<std::size_t> row_indices(matrix.rows());
    std::iota(row_indices.begin(), row_indices.end(), 0);
    bmmpy::RowWindow window = matrix.row_window(row_indices);
    std::println("Window size: {}\n", window.size());

    bmmpy::ga::GeneticAlgorithmConfig config;
    config.population_size = 300;
    config.mutation_rate = 1.0;
    config.elite_count = 3;
    config.tournament_size = 2;
    config.seed = 666;
    config.stop.max_generations = 1000;
    config.stop.max_stale_generations = 1000;
    bmmpy::ga::GeneticAlgorithm ga(config);

    std::println("Starting optimization...\n");

    using steady_clock = std::chrono::steady_clock;
    const auto opt_start = steady_clock::now();
    // auto best_individual = ga.optimize(window);

    std::size_t i = 0;
    ga.initialize(window);
    while (!ga.done()){
        ga.step();
        std::println("[ga:iter-{} best score: {}, generations: {}, evaluations: {}",
                    i++,
                    ga.best_score(),
                    ga.stats().generations,
                    ga.stats().evaluations);
    }
    auto best_individual = ga.best_individual();

    const auto opt_end = steady_clock::now();
    const auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start).count();
    std::println("opt completed in {} ms", opt_duration);

    auto best_score = ga.best_score();
    auto stats = ga.stats();

    std::println("\n=== Results ===");
    std::println("Generations: {}", stats.generations);
    std::println("Evaluations: {}", stats.evaluations);
    std::println("Best score: {}", best_score);
    std::println(
        "Improvement: {} -> {} (-{})\n", matrix.weight(), best_score, matrix.weight() - best_score);

    // print_individual(best_individual, window);

    auto result = apply_individual(matrix, window, best_individual);
    // print_matrix("\nOptimized matrix", result);

    std::string output_filename = "optimized_matrix.txt";
    try {
        result.save_text(output_filename);
        std::println("\nResult saved to '{}'", output_filename);
    } catch (const std::exception& ex) {
        std::println("\nCould not save result: {}", ex.what());
    }

    return 0;
}

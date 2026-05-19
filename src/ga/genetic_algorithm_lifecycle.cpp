#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/ga/genetic_algorithm_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <print>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::ga {

GeneticAlgorithm::GeneticAlgorithm(GeneticAlgorithmConfig config) : _config(std::move(config)) {}

void GeneticAlgorithm::initialize(const RowWindow& window) {
    _window = &window;
    _N = _window->size();
    _M = _window->cols();

    const auto init_started = internal::steady_clock::now();

    if (_N == 0 || _M == 0)
        throw std::invalid_argument("GeneticAlgorithm::initialize: empty window is not supported");

    if (_config.population_size == 0) {
        throw std::invalid_argument(
            "GeneticAlgorithm::initialize: population_size must be greater than zero");
    }

    _rng.reseed(_config.seed);

    _population.clear();
    _population.reserve(_config.population_size);

    _population.push_back(make_identity());
    const std::size_t target_heuristic_count =
        std::min<std::size_t>(_config.population_size / 4, _config.population_size - 1);

    for (std::size_t i = 0; i < target_heuristic_count; ++i)
        _population.push_back(make_heuristic());
    while (_population.size() < _config.population_size)
        _population.push_back(make_random());

    _fitnesses.resize(_population.size());
    for (std::size_t i = 0; i < _population.size(); ++i)
        _fitnesses[i] = evaluate_individual(_population[i]);

    auto best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    _best_score = *best;
    _best_individual = _population[std::distance(_fitnesses.begin(), best)];

    _no_improvement = 0;
    _stats = RunStats{};
    _stats.seed = _config.seed;
    _stats.evaluations = _population.size();
    _stats.stale_generations = 0;

    _initialized = true;
    _done = internal::should_stop(_config, _stats, _best_score);

    (void)init_started;
}

bool GeneticAlgorithm::done() const noexcept { return _done; }

std::size_t GeneticAlgorithm::generation() const noexcept { return _stats.generations; }

Individual GeneticAlgorithm::best_individual() const { return _best_individual; }

RunStats GeneticAlgorithm::stats() const { return _stats; }

std::size_t GeneticAlgorithm::best_score() const { return _best_score; }

std::vector<Individual> GeneticAlgorithm::export_migrants(std::size_t max_count) {
    if (max_count == 0 || _best_individual.empty())
        return {};

    // std::print("exporting migrant: {} {} {}\n", _best_score, _stats.generations,
    // _stats.stale_generations);
    return {_best_individual};
}

void GeneticAlgorithm::import_migrants(std::vector<Individual> migrants) {
    if (!_initialized)
        return;

    for (Individual& migrant : migrants) {
        if (migrant.size() != _N)
            continue;

        const std::size_t score = evaluate_individual(migrant);
        if (score >= _best_score)
            continue;

        _best_score = score;
        _best_individual = std::move(migrant);
        _stats.stale_generations = 0;
        _no_improvement = 0;

        auto worst = std::max_element(_fitnesses.begin(), _fitnesses.end());
        const std::size_t worst_idx =
            static_cast<std::size_t>(std::distance(_fitnesses.begin(), worst));

        _population[worst_idx] = _best_individual;
        _fitnesses[worst_idx] = score;
    }

    // std::print("import migrants: {} {} {}\n", _best_score, _stats.generations,
    // _stats.stale_generations);
    _done = internal::should_stop(_config, _stats, _best_score);
}

void GeneticAlgorithm::step() {
    const auto step_started = internal::steady_clock::now();

    if (!_initialized || _done)
        return;

    constexpr std::size_t k_restart_after = 15;
    constexpr double k_restart_probability = 0.2;
    constexpr std::size_t k_local_improvement_period = 3;

    adapt_mutation_rate();

    std::vector<Individual> new_population;
    new_population.reserve(_config.population_size);

    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    ranked.reserve(_population.size());

    for (std::size_t i = 0; i < _population.size(); ++i)
        ranked.emplace_back(_fitnesses[i], i);

    std::sort(ranked.begin(), ranked.end());

    for (std::size_t i = 0; i < _config.elite_count && i < ranked.size(); ++i)
        new_population.push_back(_population[ranked[i].second]);

    // std::uniform_real_distribution<double> probability(0.0, 1.0);

    while (new_population.size() < _config.population_size) {
        if (_no_improvement > k_restart_after && _rng.next_unit_double() < k_restart_probability) {
            new_population.push_back(make_random());
            continue;
        }

        Individual parent_a = tournament_selection();
        Individual parent_b = tournament_selection();

        Individual child = crossover(parent_a, parent_b);
        mutate(child);

        if (k_local_improvement_period != 0 &&
            (_stats.generations % k_local_improvement_period) == 0) {
            local_improvement(child);
        }

        new_population.push_back(std::move(child));
    }

    _population = std::move(new_population);

    for (std::size_t i = 0; i < _population.size(); ++i)
        _fitnesses[i] = evaluate_individual(_population[i]);

    _stats.evaluations += _population.size();
    ++_stats.generations;

    auto best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    const std::size_t best_idx = static_cast<std::size_t>(std::distance(_fitnesses.begin(), best));

    if (*best < _best_score) {
        _best_score = *best;
        _best_individual = _population[best_idx];
        _stats.stale_generations = 0;
        _no_improvement = 0;
    } else {
        ++_stats.stale_generations;
        ++_no_improvement;
    }

    _done = internal::should_stop(_config, _stats, _best_score);

    (void)step_started;
}

Individual GeneticAlgorithm::optimize(const RowWindow& window) {
    initialize(window);
    while (!done())
        step();
    return best_individual();
}

} // namespace bmmpy::ga

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

    std::println("[ga:init] N={} M={} pop={} elite={} tourn={} mut_rate={} seed={} stop.max_gen={} "
                 "stop.stale={}",
                 _N,
                 _M,
                 _config.population_size,
                 _config.elite_count,
                 _config.tournament_size,
                 _config.mutation_rate,
                 _config.seed,
                 _config.stop.max_generations.value_or(0),
                 _config.stop.max_stale_generations.value_or(0));

    if (_N == 0 || _M == 0)
        throw std::invalid_argument("GeneticAlgorithm::initialize: empty window is not supported");

    if (_config.population_size == 0) {
        throw std::invalid_argument(
            "GeneticAlgorithm::initialize: population_size must be greater than zero");
    }

    if (_config.elite_count > _config.population_size) {
        std::println("[ga:init:warn] elite_count={} exceeds population_size={}",
                     _config.elite_count,
                     _config.population_size);
    }

    _rng.seed(_config.seed);
    std::println("[ga:init] rng seeded elapsed_ms={}", internal::elapsed_ms(init_started));

    _population.clear();
    _population.reserve(_config.population_size);

    const std::size_t heuristic_slots = _config.population_size / 4;
    std::println("[ga:init] building initial population count={} heuristic_slots={}",
                 _config.population_size,
                 heuristic_slots);

    for (std::size_t i = 0; i < _config.population_size; ++i) {
        const auto individual_started = internal::steady_clock::now();
        std::println("[ga:init:individual:start] idx={} elapsed_ms={}",
                     i,
                     internal::elapsed_ms(init_started));

        _population.push_back(make_random());

        std::println("[ga:init:individual:done] idx={} score={} elapsed_ms={} local_ms={}",
                     i,
                     evaluate_individual(_population.back()),
                     internal::elapsed_ms(init_started),
                     internal::elapsed_ms(individual_started));
    }

    // Здесь можно добавить heuristic seeding, если вернёте make_heuristic().
    _fitnesses.resize(_population.size());

    std::println("[ga:init] evaluating initial population size={}", _population.size());
    for (std::size_t i = 0; i < _population.size(); ++i) {
        _fitnesses[i] = evaluate_individual(_population[i]);
        std::println("[ga:init:fitness] idx={} score={} elapsed_ms={}",
                     i,
                     _fitnesses[i],
                     internal::elapsed_ms(init_started));
    }

    auto best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    _best_score = *best;
    _best_individual = _population[std::distance(_fitnesses.begin(), best)];
    _no_improvement = 0;

    _stats = RunStats{};
    _stats.seed = _config.seed;
    _stats.evaluations = _population.size();

    _initialized = true;
    _done = internal::should_stop(_config, _stats, _best_score);

    std::println("[ga:init:done] best_score={} evaluations={} done={} elapsed_ms={}",
                 _best_score,
                 _stats.evaluations,
                 _done,
                 internal::elapsed_ms(init_started));
}

void GeneticAlgorithm::step() {
    const auto step_started = internal::steady_clock::now();

    if (!_initialized || _done)
        return;

    std::println("[ga:step:start] generation={} pop={} best={} stale={} mut_rate={:.3f}",
                 _stats.generations,
                 _population.size(),
                 _best_score,
                 _stats.stale_generations,
                 _config.mutation_rate);

    auto best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    if (*best < _best_score) {
        _best_score = *best;
        _best_individual = _population[std::distance(_fitnesses.begin(), best)];
        _stats.stale_generations = 0;
        _no_improvement = 0;
    } else {
        ++_stats.stale_generations;
        ++_no_improvement;
    }

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

    std::println("[ga:step] elites_copied={} elapsed_ms={}",
                 new_population.size(),
                 internal::elapsed_ms(step_started));

    std::uniform_real_distribution<double> probability(0.0, 1.0);

    while (new_population.size() < _config.population_size) {
        std::println("[ga:step:child:start] generation={} next_child_idx={} elapsed_ms={}",
                     _stats.generations,
                     new_population.size(),
                     internal::elapsed_ms(step_started));

        if (_no_improvement > 15 && probability(_rng) < 0.2) {
            std::println("[ga:step:child] generation={} next_child_idx={} source=random_restart",
                         _stats.generations,
                         new_population.size());
            new_population.push_back(make_random());
            continue;
        }

        const auto selection_started = internal::steady_clock::now();
        Individual parent_a = tournament_selection();
        Individual parent_b = tournament_selection();
        const auto selection_ms = internal::elapsed_ms(selection_started);

        const auto crossover_started = internal::steady_clock::now();
        Individual child = crossover(parent_a, parent_b);
        const auto crossover_ms = internal::elapsed_ms(crossover_started);

        const auto mutate_started = internal::steady_clock::now();
        mutate(child);
        const auto mutate_ms = internal::elapsed_ms(mutate_started);

        long long improve_ms = 0;

        // if (_stats.generations % 3 == 0) {
        //     const auto improve_started = internal::steady_clock::now();
        //     local_improvement(child);
        //     improve_ms = internal::elapsed_ms(improve_started);
        // }

        std::println("[ga:child] gen={} child_idx={} mutate_ms={} improve_ms={} child_score={} "
                     "crossover_ms={} selection_ms={}",
                     _stats.generations,
                     new_population.size(),
                     mutate_ms,
                     improve_ms,
                     evaluate_individual(child),
                     crossover_ms,
                     selection_ms);

        new_population.push_back(std::move(child));
    }

    _population = std::move(new_population);

    for (std::size_t i = 0; i < _population.size(); ++i)
        _fitnesses[i] = evaluate_individual(_population[i]);

    _stats.evaluations += _population.size();
    ++_stats.generations;

    best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    if (*best < _best_score) {
        _best_score = *best;
        _best_individual = _population[std::distance(_fitnesses.begin(), best)];
    }

    std::println("[ga:step] gen={} best={} stale={} mut_rate={:.3f} evals={} step_ms={}",
                 _stats.generations,
                 _best_score,
                 _stats.stale_generations,
                 _config.mutation_rate,
                 _stats.evaluations,
                 internal::elapsed_ms(step_started));

    _done = internal::should_stop(_config, _stats, _best_score);

    std::println("[ga:step:done] generation={} done={} elapsed_ms={}",
                 _stats.generations,
                 _done,
                 internal::elapsed_ms(step_started));
}

bool GeneticAlgorithm::done() const noexcept { return _done; }

std::size_t GeneticAlgorithm::generation() const noexcept { return _stats.generations; }

Individual GeneticAlgorithm::best_individual() const { return _best_individual; }

RunStats GeneticAlgorithm::stats() const { return _stats; }

std::size_t GeneticAlgorithm::best_score() const { return _best_score; }

std::vector<Individual> GeneticAlgorithm::export_migrants(std::size_t max_count) {
    if (max_count == 0 || _best_individual.empty())
        return {};
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
        const std::size_t worst_idx = static_cast<std::size_t>(
            std::distance(_fitnesses.begin(), worst));

        _population[worst_idx] = _best_individual;
        _fitnesses[worst_idx] = score;
    }

    _done = internal::should_stop(_config, _stats, _best_score);
}

Individual GeneticAlgorithm::optimize(const RowWindow& window) {
    initialize(window);
    while (!done())
        step();
    return best_individual();
}

} // namespace bmmpy::ga

#include "bmmpy/ga/genetic_algorithm.hpp"

#include "bmmpy/core/detail/xor_basis.hpp"
#include "bmmpy/types/candidate.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <print>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::ga {
namespace {

using steady_clock = std::chrono::steady_clock;

long long elapsed_ms(const steady_clock::time_point started) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - started)
        .count();
}

bool should_stop(const GeneticAlgorithmConfig& cfg, const RunStats& st, std::size_t best) {
    if (cfg.stop.max_generations && st.generations >= *cfg.stop.max_generations)
        return true;
    if (cfg.stop.max_stale_generations && st.stale_generations >= *cfg.stop.max_stale_generations)
        return true;
    if (cfg.stop.target_total_weight && best <= *cfg.stop.target_total_weight)
        return true;
    return false;
}

} // namespace

GeneticAlgorithm::GeneticAlgorithm(GeneticAlgorithmConfig config) : _config(std::move(config)) {}

std::size_t GeneticAlgorithm::evaluate_individual(const Individual& ind) const {
    std::size_t total = 0;
    for (const Candidate& c : ind)
        total += c.weight;
    return total;
}

namespace {

std::uint32_t evaluate_candidate_weight(const RowWindow& window,
                                        const std::size_t row_count,
                                        const std::size_t col_count,
                                        const Candidate& candidate) {
    std::vector<bool> row(col_count, false);
    for (std::size_t r = 0; r < row_count; ++r) {
        if (candidate.has_row(r)) {
            for (std::size_t col = 0; col < col_count; ++col) {
                row[col] = row[col] != window.get(r, col);
            }
        }
    }

    return static_cast<std::uint32_t>(std::count(row.begin(), row.end(), true));
}

} // namespace

void GeneticAlgorithm::recalc_all_weights(Individual& ind) {
    const auto recalc_started = steady_clock::now();
    // std::println("[ga:recalc:start] candidates={} rows={} cols={}", ind.size(), _N, _M);

    for (Candidate& c : ind) {
        c.weight = evaluate_candidate_weight(*_window, _N, _M, c);

        const std::size_t candidate_idx = static_cast<std::size_t>(&c - ind.data());
        // std::println("[ga:recalc:candidate] idx={} weight={} elapsed_ms={}",
        //              candidate_idx,
        //              c.weight,
        //              elapsed_ms(recalc_started));
    }

    // std::println(
    //     "[ga:recalc:done] candidates={} elapsed_ms={}", ind.size(), elapsed_ms(recalc_started));
}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/*
bool GeneticAlgorithm::is_non_singular(const Individual& ind) const {
    if (ind.size() != _N) return false;

    std::vector<std::uint64_t> rows;
    rows.reserve(_N);
    for (const Candidate& c : ind) {
        std::uint64_t val = 0;
        for (std::size_t bit = 0; bit < _N; ++bit) {
            if (c.has_row(bit)) val |= (std::uint64_t{1} << bit);
        }
        rows.push_back(val);
    }

    std::size_t rank = 0;
    for (std::size_t col = 0; col < _N && rank < _N; ++col) {
        std::size_t pivot = rank;
        while (pivot < _N && ((rows[pivot] >> col) & 1) == 0) ++pivot;
        if (pivot == _N) continue;
        if (pivot != rank) std::swap(rows[rank], rows[pivot]);
        for (std::size_t r = rank + 1; r < _N; ++r) {
            if ((rows[r] >> col) & 1) rows[r] ^= rows[rank];
        }
        ++rank;
    }
    return rank == _N;
}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void GeneticAlgorithm::ensure_non_singular(Individual& ind) {
    if (!is_non_singular(ind)) {
        ind = make_identity();
    }
}
*/
void GeneticAlgorithm::adapt_mutation_rate() {
    if (_no_improvement > 10)
        _config.mutation_rate = std::min(0.8, _config.mutation_rate * 1.5);
    else if (_no_improvement > 5)
        _config.mutation_rate = std::min(0.6, _config.mutation_rate * 1.3);
    else
        _config.mutation_rate = std::max(0.05, _config.mutation_rate * 0.95);
}

void GeneticAlgorithm::initialize(const RowWindow& window) {
    _window = &window;
    _N = _window->size();
    _M = _window->cols();

    const auto init_started = steady_clock::now();

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

    if (_N == 0 || _M == 0) {
        throw std::invalid_argument("GeneticAlgorithm::initialize: empty window is not supported");
    }

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
    std::println("[ga:init] rng seeded elapsed_ms={}", elapsed_ms(init_started));

    _population.clear();
    _population.reserve(_config.population_size);
    const std::size_t hc = _config.population_size / 4;
    std::println("[ga:init] building initial population count={} heuristic_slots={}",
                 _config.population_size,
                 hc);
    for (std::size_t i = 0; i < _config.population_size; ++i) {
        const auto individual_started = steady_clock::now();
        std::println(
            "[ga:init:individual:start] idx={} elapsed_ms={}", i, elapsed_ms(init_started));
        _population.push_back(make_random());
        std::println("[ga:init:individual:done] idx={} score={} elapsed_ms={} local_ms={}",
                     i,
                     evaluate_individual(_population.back()),
                     elapsed_ms(init_started),
                     elapsed_ms(individual_started));
    }

    // for (std::size_t i = 0; i < hc; ++i)
    //_population.push_back(make_heuristic());

    _fitnesses.resize(_population.size());
    std::println("[ga:init] evaluating initial population size={}", _population.size());
    for (std::size_t i = 0; i < _population.size(); ++i) {
        _fitnesses[i] = evaluate_individual(_population[i]);
        std::println("[ga:init:fitness] idx={} score={} elapsed_ms={}",
                     i,
                     _fitnesses[i],
                     elapsed_ms(init_started));
    }

    auto best = std::min_element(_fitnesses.begin(), _fitnesses.end());
    _best_score = *best;
    _best_individual = _population[std::distance(_fitnesses.begin(), best)];
    _no_improvement = 0;

    _stats = RunStats{};
    _stats.seed = _config.seed;
    _stats.evaluations = _population.size();

    _initialized = true;
    _done = should_stop(_config, _stats, _best_score);
    std::println("[ga:init:done] best_score={} evaluations={} done={} elapsed_ms={}",
                 _best_score,
                 _stats.evaluations,
                 _done,
                 elapsed_ms(init_started));
}

void GeneticAlgorithm::step() {
    const auto step_started = steady_clock::now();

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

    std::vector<Individual> new_pop;
    new_pop.reserve(_config.population_size);

    std::vector<std::pair<std::size_t, std::size_t>> ranked;
    for (std::size_t i = 0; i < _population.size(); ++i)
        ranked.emplace_back(_fitnesses[i], i);
    std::sort(ranked.begin(), ranked.end());
    for (std::size_t i = 0; i < _config.elite_count && i < ranked.size(); ++i)
        new_pop.push_back(_population[ranked[i].second]);
    std::println(
        "[ga:step] elites_copied={} elapsed_ms={}", new_pop.size(), elapsed_ms(step_started));

    std::uniform_real_distribution<double> prob(0, 1);
    while (new_pop.size() < _config.population_size) {
        std::println("[ga:step:child:start] generation={} next_child_idx={} elapsed_ms={}",
                     _stats.generations,
                     new_pop.size(),
                     elapsed_ms(step_started));

        if (_no_improvement > 15 && prob(_rng) < 0.2) {
            std::println("[ga:step:child] generation={} next_child_idx={} source=random_restart",
                         _stats.generations,
                         new_pop.size());
            new_pop.push_back(make_random());
            continue;
        }

        const auto selection_started = steady_clock::now();
        Individual p1 = tournament_selection();
        Individual p2 = tournament_selection();
        const auto selection_ms = elapsed_ms(selection_started);

        const auto crossover_started = steady_clock::now();
        auto child = crossover(p1, p2);
        const auto crossover_ms = elapsed_ms(crossover_started);

        const auto mutate_started = steady_clock::now();
        mutate(child);
        const auto mutate_ms = elapsed_ms(mutate_started);
        long long improve_ms = 0;
        // if (_stats.generations % 3 == 0) {
        //     const auto improve_started = steady_clock::now();
        //     local_improvement(child);
        //     improve_ms = elapsed_ms(improve_started);
        // }

        std::println("[ga:child] gen={} child_idx={} mutate_ms={} improve_ms={} child_score={} "
                     "crossover_ms={} selection_ms={}",
                     _stats.generations,
                     new_pop.size(),
                     mutate_ms,
                     improve_ms,
                     evaluate_individual(child),
                     crossover_ms,
                     selection_ms);
        new_pop.push_back(std::move(child));
    }

    _population = std::move(new_pop);
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
                 elapsed_ms(step_started));

    _done = should_stop(_config, _stats, _best_score);
    std::println("[ga:step:done] generation={} done={} elapsed_ms={}",
                 _stats.generations,
                 _done,
                 elapsed_ms(step_started));
}

Individual GeneticAlgorithm::tournament_selection() {
    std::uniform_int_distribution<std::size_t> idx(0, _population.size() - 1);
    std::size_t best_idx = idx(_rng);
    for (std::size_t i = 1; i < _config.tournament_size; ++i) {
        std::size_t cand = idx(_rng);
        if (_fitnesses[cand] < _fitnesses[best_idx])
            best_idx = cand;
    }
    return _population[best_idx];
}
/*
Individual GeneticAlgorithm::crossover(const Individual& p1, const Individual& p2) {
    std::vector<Candidate> pool;
    for (const Candidate& c : p1) pool.push_back(c);
    for (const Candidate& c : p2) pool.push_back(c);

    // сортировка по весу (сначала лучшие)
    std::sort(pool.begin(), pool.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.weight < b.weight;
              });

    // ребенок
    Individual child;
    child.reserve(_N);

    // Добавление кандидатов от родителей с проверкой на невырожденность
    for (const Candidate& c : pool) {
        if (child.size() == _N) break;

        child.push_back(c);

        // Если набрано N строк и матрица вырождена — откат
        if (child.size() == _N && !is_non_singular(child)) {
            child.pop_back();
        }
    }

    // Добивка единичными кандидатами недостающие строки
    for (std::size_t i = 0; i < _N && child.size() < _N; ++i) {
        child.push_back(Candidate::make_unit(
            _N, i, static_cast<std::uint32_t>(_window->row_popcount(i))));

        // Если набрано N строк и матрица вырождена — откат
        if (child.size() == _N && !is_non_singular(child)) {
            child.pop_back();
        }
    }

    if (child.size() != _N) {
        child = make_identity();
    }
    recalc_all_weights(child);
    return child;
}
*/

inline Individual GeneticAlgorithm::crossover(const Individual& a, const Individual& b) {
    const auto crossover_started = steady_clock::now();
    std::println("[ga:crossover:start] lhs={} rhs={} elapsed_ms=0", a.size(), b.size());

    std::vector<const Candidate*> pool;
    pool.reserve(a.size() + b.size());
    for (const auto& c : a)
        pool.push_back(&c);
    for (const auto& c : b)
        pool.push_back(&c);

    std::stable_sort(pool.begin(), pool.end(), [](const Candidate* lhs, const Candidate* rhs) {
        return lhs->mask_popcount() < rhs->mask_popcount();
    });

    const std::size_t bit_width = a.size();
    detail::PivotBasis pivot(bit_width);
    Individual child;

    for (const Candidate* candidate : pool) {
        if (pivot.try_insert(candidate->mask))
            child.push_back(*candidate);
        if (child.size() == bit_width)
            break;
    }

    for (std::size_t i = 0; i < bit_width && child.size() < bit_width; ++i) {
        auto unit = detail::make_unit_mask_words(bit_width, i);
        if (pivot.try_insert(unit))
            child.push_back(Candidate(std::move(unit), 1));
    }

    std::println("[ga:crossover:done] child_size={} rank={} elapsed_ms={}",
                 child.size(),
                 pivot.rank(),
                 elapsed_ms(crossover_started));

    return child;
}

void GeneticAlgorithm::mutate(Individual& ind) {
    const auto mutate_started = steady_clock::now();
    std::size_t n = std::max<std::size_t>(1, _N * _config.mutation_rate);
    std::uniform_int_distribution<std::size_t> rd(0, _N - 1);
    std::println("[ga:mutate:start] n={} rows={} elapsed_ms=0", n, _N);

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t i = rd(_rng), j = rd(_rng);
        if (i == j)
            continue;

        std::println("[ga:mutate:iter:start] iter={} i={} j={} elapsed_ms={}",
                     k,
                     i,
                     j,
                     elapsed_ms(mutate_started));

        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];

        recalc_all_weights(ind);
        std::println("[ga:mutate:iter:done] iter={} elapsed_ms={}", k, elapsed_ms(mutate_started));
    }

    std::println("[ga:mutate:done] n={} elapsed_ms={}", n, elapsed_ms(mutate_started));
}

void GeneticAlgorithm::local_improvement(Individual& ind) {
    const auto improve_started = steady_clock::now();
    std::println("[ga:local:start] rows={} elapsed_ms=0", _N);

    for (std::size_t iter = 0; iter < 10; ++iter) {
        bool improved = false;
        std::println(
            "[ga:local:iter:start] iter={} elapsed_ms={}", iter, elapsed_ms(improve_started));
        for (std::size_t i = 0; i < _N; ++i) {
            for (std::size_t j = 0; j < _N; ++j) {
                if (i == j)
                    continue;

                // std::println("[ga:local:try] iter={} i={} j={} elapsed_ms={}",
                //              iter,
                //              i,
                //              j,
                //              elapsed_ms(improve_started));

                Candidate::mask_type old_mask = ind[i].mask;
                std::uint32_t old_w = ind[i].weight;

                for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
                    ind[i].mask[w] ^= ind[j].mask[w];

                const std::uint32_t new_weight =
                    evaluate_candidate_weight(*_window, _N, _M, ind[i]);
                if (new_weight < old_w) {
                    ind[i].weight = new_weight;
                    std::println(
                        "[ga:local:improved] iter={} i={} j={} old_w={} new_w={} elapsed_ms={}",
                        iter,
                        i,
                        j,
                        old_w,
                        new_weight,
                        elapsed_ms(improve_started));
                    improved = true;
                    break;
                }

                ind[i].mask = std::move(old_mask);
                ind[i].weight = old_w;
                // std::println("[ga:local:revert] iter={} i={} j={} elapsed_ms={}",
                //              iter,
                //              i,
                //              j,
                //              elapsed_ms(improve_started));
            }
            if (improved)
                break;
        }
        if (!improved)
            break;
    }

    std::println("[ga:local:done] elapsed_ms={}", elapsed_ms(improve_started));
}

Individual GeneticAlgorithm::make_identity() const {
    Individual ind;
    ind.reserve(_N);
    for (std::size_t i = 0; i < _N; ++i)
        ind.push_back(Candidate::make_unit(_N, i, _window->row_popcount(i)));
    return ind;
}

Individual GeneticAlgorithm::make_random() {
    const auto random_started = steady_clock::now();
    std::println("[ga:random:start] rows={} cols={} elapsed_ms=0", _N, _M);

    Individual ind = make_identity();
    std::uniform_int_distribution<std::size_t> rd(0, _N - 1);
    for (std::size_t k = 0; k < _N * 3; ++k) {
        std::size_t i = rd(_rng), j = rd(_rng);
        if (i == j)
            continue;

        std::println(
            "[ga:random:xor] iter={} i={} j={} elapsed_ms={}", k, i, j, elapsed_ms(random_started));

        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];
    }

    std::println("[ga:random] before_recalc elapsed_ms={}", elapsed_ms(random_started));
    recalc_all_weights(ind);
    std::println("[ga:random:done] score={} elapsed_ms={}",
                 evaluate_individual(ind),
                 elapsed_ms(random_started));
    return ind;
}
/*
Individual GeneticAlgorithm::make_heuristic() {
    Individual ind = make_identity();
    std::vector<std::pair<std::size_t, std::size_t>> rw;
    for (std::size_t i = 0; i < _N; ++i) rw.emplace_back(_window->row_popcount(i), i);
    std::sort(rw.begin(), rw.end(), std::greater<>{});
    for (std::size_t k = 0; k < std::min<std::size_t>(5, _N); ++k) {
        std::size_t idx = rw[k].second;
        for (std::size_t j = 0; j < _N; ++j) {
            if (idx == j) continue;
            Candidate::mask_type old_mask = ind[idx].mask;
            std::uint32_t old_w = ind[idx].weight;
            for (std::size_t w = 0; w < ind[idx].mask.size(); ++w)
                ind[idx].mask[w] ^= ind[j].mask[w];
            if (!is_non_singular(ind)) {
                ind[idx].mask = std::move(old_mask);
                ind[idx].weight = old_w;
                continue;
            }
            recalc_all_weights(ind);
            if (ind[idx].weight < old_w) break;
            ind[idx].mask = std::move(old_mask);
            ind[idx].weight = old_w;
        }
    }
    return ind;
}
*/
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
    for (Individual& m : migrants) {
        if (m.size() != _N)
            continue;
        // if (!is_non_singular(m)) continue;  // пропускаем вырожденных мигрантов
        std::size_t sc = evaluate_individual(m);
        if (sc < _best_score) {
            _best_score = sc;
            _best_individual = std::move(m);
            _stats.stale_generations = 0;
            _no_improvement = 0;
            auto worst = std::max_element(_fitnesses.begin(), _fitnesses.end());
            _population[std::distance(_fitnesses.begin(), worst)] = _best_individual;
            _fitnesses[std::distance(_fitnesses.begin(), worst)] = sc;
        }
    }
    _done = should_stop(_config, _stats, _best_score);
}

Individual GeneticAlgorithm::optimize(const RowWindow& window) {
    initialize(window);
    while (!done())
        step();
    return best_individual();
}

} // namespace bmmpy::ga
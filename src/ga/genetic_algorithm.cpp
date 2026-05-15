#include "bmmpy/ga/genetic_algorithm.hpp"
#include "bmmpy/types/candidate.hpp"
#include "bmmpy/core/detail/xor_basis.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::ga {
namespace {

bool should_stop(const GeneticAlgorithmConfig& cfg, const RunStats& st, std::size_t best) {
    if (cfg.stop.max_generations && st.generations >= *cfg.stop.max_generations) return true;
    if (cfg.stop.max_stale_generations && st.stale_generations >= *cfg.stop.max_stale_generations) return true;
    if (cfg.stop.target_total_weight && best <= *cfg.stop.target_total_weight) return true;
    return false;
}

} // namespace

GeneticAlgorithm::GeneticAlgorithm(GeneticAlgorithmConfig config) : _config(std::move(config)) {}

std::size_t GeneticAlgorithm::evaluate_individual(const Individual& ind) const {
    std::size_t total = 0;
    for (const Candidate& c : ind) total += c.weight;
    return total;
}

void GeneticAlgorithm::recalc_all_weights(Individual& ind) {
    for (Candidate& c : ind) {
        std::vector<bool> row(_M, false);
        for (std::size_t r = 0; r < _N; ++r) {
            if (c.has_row(r)) {
                for (std::size_t col = 0; col < _M; ++col) {
                    row[col] = row[col] != _window->get(r, col);
                }
            }
        }
        c.weight = static_cast<std::uint32_t>(std::count(row.begin(), row.end(), true));
    }
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

    _rng.seed(_config.seed);

    _population.clear();
    _population.reserve(_config.population_size);
    const std::size_t hc = _config.population_size / 4;
    for (std::size_t i = 0; i < _config.population_size; ++i)
        _population.push_back(make_random());

    //for (std::size_t i = 0; i < hc; ++i)
        //_population.push_back(make_heuristic());

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

    _initialized = true;
    _done = should_stop(_config, _stats, _best_score);
}

void GeneticAlgorithm::step() {
    if (!_initialized || _done) return;

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

    std::uniform_real_distribution<double> prob(0, 1);
    while (new_pop.size() < _config.population_size) {
        if (_no_improvement > 15 && prob(_rng) < 0.2) {
            new_pop.push_back(make_random());
            continue;
        }

        Individual p1 = tournament_selection();
        Individual p2 = tournament_selection();
        auto child = crossover(p1, p2);
        mutate(child);
        if (_stats.generations % 3 == 0) local_improvement(child);
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

    _done = should_stop(_config, _stats, _best_score);
}

Individual GeneticAlgorithm::tournament_selection() {
    std::uniform_int_distribution<std::size_t> idx(0, _population.size() - 1);
    std::size_t best_idx = idx(_rng);
    for (std::size_t i = 1; i < _config.tournament_size; ++i) {
        std::size_t cand = idx(_rng);
        if (_fitnesses[cand] < _fitnesses[best_idx]) best_idx = cand;
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
    std::vector<const Candidate*> pool;
    pool.reserve(a.size() + b.size());
    for (const auto& c : a) pool.push_back(&c);
    for (const auto& c : b) pool.push_back(&c);

    std::stable_sort(pool.begin(), pool.end(),
        [](const Candidate* lhs, const Candidate* rhs) {
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

    return child;
}



void GeneticAlgorithm::mutate(Individual& ind) {
    std::size_t n = std::max<std::size_t>(1, _N * _config.mutation_rate);
    std::uniform_int_distribution<std::size_t> rd(0, _N - 1);
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t i = rd(_rng), j = rd(_rng);
        if (i == j) continue;
        
        Candidate::mask_type old = ind[i].mask;
        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];
        
        recalc_all_weights(ind);
    }
}

void GeneticAlgorithm::local_improvement(Individual& ind) {
    for (std::size_t iter = 0; iter < 10; ++iter) {
        bool improved = false;
        for (std::size_t i = 0; i < _N; ++i) {
            for (std::size_t j = 0; j < _N; ++j) {
                if (i == j) continue;
                
                Candidate::mask_type old_mask = ind[i].mask;
                std::uint32_t old_w = ind[i].weight;
                
                for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
                    ind[i].mask[w] ^= ind[j].mask[w];
                
                recalc_all_weights(ind);
                if (ind[i].weight < old_w) { improved = true; break; }
                
                ind[i].mask = std::move(old_mask);
                ind[i].weight = old_w;
            }
            if (improved) break;
        }
        if (!improved) break;
    }
}

Individual GeneticAlgorithm::make_identity() const {
    Individual ind;
    ind.reserve(_N);
    for (std::size_t i = 0; i < _N; ++i)
        ind.push_back(Candidate::make_unit(_N, i, _window->row_popcount(i)));
    return ind;
}

Individual GeneticAlgorithm::make_random() {
    Individual ind = make_identity();
    std::uniform_int_distribution<std::size_t> rd(0, _N - 1);
    for (std::size_t k = 0; k < _N * 3; ++k) {
        std::size_t i = rd(_rng), j = rd(_rng);
        if (i == j) continue;
        Candidate::mask_type old = ind[i].mask;
        for (std::size_t w = 0; w < ind[i].mask.size(); ++w)
            ind[i].mask[w] ^= ind[j].mask[w];
        ind[i].mask = std::move(old); continue;
    }
    recalc_all_weights(ind);
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
    if (max_count == 0 || _best_individual.empty()) return {};
    return {_best_individual};
}

void GeneticAlgorithm::import_migrants(std::vector<Individual> migrants) {
    if (!_initialized) return;
    for (Individual& m : migrants) {
        if (m.size() != _N) continue;
        //if (!is_non_singular(m)) continue;  // пропускаем вырожденных мигрантов
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
    while (!done()) step();
    return best_individual();
}

} // namespace bmmpy::ga
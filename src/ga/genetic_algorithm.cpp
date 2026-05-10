#include "bmmpy/ga/genetic_algorithm.hpp"

#include "bmmpy/types/candidate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::ga {
namespace {

// Individual это просто вектор кандидатов (std::vector<::bmmpy::Candidate>), а кандидат - просто
// маска. То есть по сути это и есть матрица перехода, только с ней работать удобнее, чем с
// настоящей матрицей.
// В нашем случае, счет/score - это просто сумма весов кандидатов.
// Если совсем коротко: сейчас “лучший” означает “с наименьшей суммой весов кандидатов”, но по факту
// реализуется пока только в импорте мигрантов и на уровни модели островов.
std::size_t score_individual(const Individual& individual) {
    std::size_t total = 0;
    for (const Candidate& candidate : individual) {
        total += candidate.weight;
    }
    return total;
}

// строит тривиальное начальное решение состоящее из всех единичных кандидатов, т.е. каждого
// кандидата, который выбирает ровно одну строку.
// Первая строка берет только первую строку окна
// Вторая строка только вторую
// И так далее
Individual make_identity_individual(const RowWindow& window) {
    Individual individual;
    individual.reserve(window.size());

    for (std::size_t row = 0; row < window.size(); ++row) {
        individual.push_back(Candidate::make_unit(
            window.size(), row, static_cast<std::uint32_t>(window.row_popcount(row))));
    }

    return individual;
}

// эта штука работает с помощью StopCriteria из GeneticAlgorithmConfig. Сейчас там есть три
// критерия: максимальное количество поколений, максимальное количество "застойных" поколений и
// достижение целевого веса. Если любой из этих критериев выполняется, алгоритм останавливается.
// Критерия по времени там пока нет, но их (и другие) можно добавить в будущем.
bool should_stop(const GeneticAlgorithmConfig& config,
                 const RunStats& stats,
                 const std::size_t best_score) {
    if (config.stop.max_generations && stats.generations >= *config.stop.max_generations) {
        return true;
    }

    if (config.stop.max_stale_generations &&
        stats.stale_generations >= *config.stop.max_stale_generations) {
        return true;
    }

    if (config.stop.target_total_weight && best_score <= *config.stop.target_total_weight) {
        return true;
    }

    return false;
}

} // namespace

GeneticAlgorithm::GeneticAlgorithm(GeneticAlgorithmConfig config) : _config(std::move(config)) {}

// в конкретном случае ГА, под окном понимается вся матрица, так как мы не рассматриваем оконную
// оптимизацию для ГА.
void GeneticAlgorithm::initialize(const RowWindow& window) {
    _best_individual = make_identity_individual(window);
    _best_score = score_individual(_best_individual);

    _stats = RunStats{};
    _stats.seed = _config.seed;

    _initialized = true;
    _done = should_stop(_config, _stats, _best_score);
}

// это сам непосредственный шаг эволюции на одно поколение (генерацию), который включает в себя все
// что захочешь (мутацию, селекцию, кроссовер и так далее). В текущей заглушке делается только
// инкремент статистики и проверка на остановку.
void GeneticAlgorithm::step() {
    if (!_initialized) {
        throw std::logic_error("GeneticAlgorithm::step: algorithm must be initialized first");
    }

    if (_done) {
        return;
    }

    ++_stats.generations;
    ++_stats.evaluations;
    ++_stats.stale_generations;

    _done = should_stop(_config, _stats, _best_score);
}

// эта штука выставляется в true, когда алгоритм достиг одного из критериев остановки, определенных,
// опять же, в GeneticAlgorithmConfig.
bool GeneticAlgorithm::done() const noexcept { return _done; }

std::size_t GeneticAlgorithm::generation() const noexcept { return _stats.generations; }

//
Individual GeneticAlgorithm::best_individual() const { return _best_individual; }

RunStats GeneticAlgorithm::stats() const { return _stats; }

// это для островной модели
std::vector<Individual> GeneticAlgorithm::export_migrants(const std::size_t max_count) {
    if (max_count == 0 || _best_individual.empty()) {
        return {};
    }

    return std::vector<Individual>{_best_individual};
}

// и это тоже. Импорт и экспорт должен быть просто реализован, островная модель сама решает когда и
// кого импортировать и экспортировать, а алгоритмы просто предоставляют интерфейс для этого.
void GeneticAlgorithm::import_migrants(std::vector<Individual> migrants) {
    for (Individual& migrant : migrants) {
        const std::size_t migrant_score = score_individual(migrant);
        if (migrant_score < _best_score) {
            _best_score = migrant_score;
            _best_individual = std::move(migrant);
            _stats.stale_generations = 0;
        }
    }

    _done = should_stop(_config, _stats, _best_score);
}

std::size_t GeneticAlgorithm::best_score() const { return _best_score; }

// это просто сахар для оптимизации, который позволяет запустить алгоритм от начала и до конца за
// один вызов.
Individual GeneticAlgorithm::optimize(const RowWindow& window) {
    initialize(window);
    while (!done()) {
        step();
    }
    return best_individual();
}

} // namespace bmmpy::ga
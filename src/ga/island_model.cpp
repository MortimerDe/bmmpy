#include "bmmpy/ga/island_model.hpp"

#include "bmmpy/ga/island.hpp"

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::ga {

class IslandModel::Impl {
public:
    Impl(IslandModelConfig config_value,
         AlgorithmFactory algorithm_factory_value,
         std::unique_ptr<migration::Channel> migration_channel_value)
        : config(std::move(config_value)), algorithm_factory(std::move(algorithm_factory_value)),
          migration_channel(std::move(migration_channel_value)) {
        if (!migration_channel) {
            throw std::invalid_argument("IslandModel: migration channel must not be null");
        }

        if (!algorithm_factory) {
            throw std::invalid_argument("IslandModel: algorithm factory must not be empty");
        }
    }

    void initialize(const RowWindow& window) {
        if (running.load(std::memory_order_acquire)) {
            throw std::logic_error("IslandModel::initialize: model is already running");
        }

        islands.clear();
        islands.reserve(config.islands.size());

        migration_channel->clear();

        for (const IslandSpec& spec : config.islands) {
            std::unique_ptr<Algorithm> algorithm = algorithm_factory(spec);
            if (!algorithm) {
                throw std::logic_error("IslandModel::initialize: algorithm factory returned null");
            }

            islands.emplace_back(spec, std::move(algorithm), *migration_channel);
            islands.back().initialize(window);
        }

        stop_requested.store(false, std::memory_order_release);
        running.store(false, std::memory_order_release);
        initialized = true;
    }

    void start() {
        if (!initialized) {
            throw std::logic_error("IslandModel::start: model must be initialized first");
        }

        bool expected = false;
        if (!running.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            throw std::logic_error("IslandModel::start: model is already running");
        }

        stop_requested.store(false, std::memory_order_release);

        for (Island& island : islands) {
            island.start();
        }
    }

    void request_stop() noexcept {
        stop_requested.store(true, std::memory_order_release);
        migration_channel->close();

        for (Island& island : islands) {
            island.request_stop();
        }
    }

    void wait() {
        for (Island& island : islands) {
            island.wait();
        }

        running.store(false, std::memory_order_release);
    }

    bool is_running() const noexcept { return running.load(std::memory_order_acquire); }

    bool is_stop_requested() const noexcept {
        return stop_requested.load(std::memory_order_acquire);
    }

    Individual best_individual() const {
        std::size_t best_score = std::numeric_limits<std::size_t>::max();
        Individual best;

        for (const Island& island : islands) {
            const IslandSnapshot snap = island.snapshot();
            const std::size_t score = score_of(snap.best_individual);

            if (score < best_score) {
                best_score = score;
                best = snap.best_individual;
            }
        }

        return best;
    }

    IslandModelSnapshot snapshot() const {
        IslandModelSnapshot model_snapshot;
        model_snapshot.running = running.load(std::memory_order_acquire);
        model_snapshot.stop_requested = stop_requested.load(std::memory_order_acquire);
        model_snapshot.island_count = islands.size();

        std::size_t best_score = std::numeric_limits<std::size_t>::max();

        model_snapshot.islands.reserve(islands.size());

        for (const Island& island : islands) {
            IslandSnapshot island_snapshot = island.snapshot();
            model_snapshot.total_generations += island_snapshot.stats.generations;

            const std::size_t score = score_of(island_snapshot.best_individual);
            if (score < best_score) {
                best_score = score;
                model_snapshot.best_individual = island_snapshot.best_individual;
            }

            model_snapshot.islands.push_back(std::move(island_snapshot));
        }

        return model_snapshot;
    }

    Individual run_to_completion(const RowWindow& window) {
        initialize(window);
        start();
        wait();
        return best_individual();
    }

private:
    static std::size_t score_of(const Individual& individual) {
        std::size_t total = 0;
        for (const Candidate& candidate : individual) {
            total += candidate.weight;
        }
        return total;
    }

    IslandModelConfig config;
    AlgorithmFactory algorithm_factory;
    std::unique_ptr<migration::Channel> migration_channel;
    std::vector<Island> islands;

    bool initialized = false;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
};

IslandModel::IslandModel(IslandModelConfig config,
                         AlgorithmFactory algorithm_factory,
                         std::unique_ptr<migration::Channel> migration_channel)
    : _impl(std::make_unique<Impl>(
          std::move(config), std::move(algorithm_factory), std::move(migration_channel))) {}

IslandModel::~IslandModel() = default;

IslandModel::IslandModel(IslandModel&&) noexcept = default;
IslandModel& IslandModel::operator=(IslandModel&&) noexcept = default;

void IslandModel::initialize(const RowWindow& window) { _impl->initialize(window); }

void IslandModel::start() { _impl->start(); }

void IslandModel::request_stop() noexcept { _impl->request_stop(); }

void IslandModel::wait() { _impl->wait(); }

bool IslandModel::running() const noexcept { return _impl->is_running(); }

bool IslandModel::stop_requested() const noexcept { return _impl->is_stop_requested(); }

Individual IslandModel::best_individual() const { return _impl->best_individual(); }

IslandModelSnapshot IslandModel::snapshot() const { return _impl->snapshot(); }

Individual IslandModel::run_to_completion(const RowWindow& window) {
    return _impl->run_to_completion(window);
}

} // namespace bmmpy::ga
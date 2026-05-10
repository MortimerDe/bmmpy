#include "bmmpy/ga/island.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace bmmpy::ga {
namespace {

constexpr auto k_idle_backoff = std::chrono::milliseconds(1);

} // namespace

class Island::Impl {
public:
    Impl(IslandSpec spec_value,
         std::unique_ptr<Algorithm> algorithm_value,
         migration::Channel& migration_channel_value)
        : spec(std::move(spec_value)), algorithm(std::move(algorithm_value)),
          migration_channel(migration_channel_value) {
        if (!algorithm) {
            throw std::invalid_argument("Island: algorithm must not be null");
        }
    }

    ~Impl() {
        request_stop();
        join_noexcept();
    }

    void initialize(const RowWindow& window) {
        if (running.load(std::memory_order_acquire)) {
            throw std::logic_error("Island::initialize: island is already running");
        }

        std::lock_guard<std::mutex> lock(state_mutex);

        if (worker.joinable()) {
            throw std::logic_error("Island::initialize: worker thread still joinable");
        }

        materialized_window = window.materialize();
        owned_rows.resize(materialized_window.rows());
        std::iota(owned_rows.begin(), owned_rows.end(), std::size_t{0});

        RowWindow owned_window(materialized_window, owned_rows);
        algorithm->initialize(owned_window);

        initialized = true;
        stop_requested.store(false, std::memory_order_release);
        finished.store(false, std::memory_order_release);
        running.store(false, std::memory_order_release);

        failure = nullptr;
        failed.store(false, std::memory_order_release);
    }

    void start() {
        if (!initialized) {
            throw std::logic_error("Island::start: island must be initialized first");
        }

        bool expected = false;
        if (!running.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            throw std::logic_error("Island::start: island is already running");
        }

        stop_requested.store(false, std::memory_order_release);
        finished.store(false, std::memory_order_release);

        worker = std::thread([this]() { loop(); });
    }

    void request_stop() noexcept { stop_requested.store(true, std::memory_order_release); }

    void wait() {
        if (worker.joinable()) {
            worker.join();
        }

        rethrow_if_failed();
    }

    bool is_running() const noexcept { return running.load(std::memory_order_acquire); }

    bool is_stop_requested() const noexcept {
        return stop_requested.load(std::memory_order_acquire);
    }

    bool is_finished() const noexcept { return finished.load(std::memory_order_acquire); }

    Individual best_individual() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return algorithm->best_individual();
    }

    RunStats stats() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return algorithm->stats();
    }

    IslandSnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        return IslandSnapshot{
            spec.island_id,
            running.load(std::memory_order_acquire),
            stop_requested.load(std::memory_order_acquire),
            finished.load(std::memory_order_acquire),
            algorithm->stats(),
            algorithm->best_score(),
            algorithm->best_individual(),
        };
    }

    const char* name() const noexcept { return algorithm->name(); }

    std::size_t island_id() const noexcept { return spec.island_id; }

private:
    void join_noexcept() noexcept {
        try {
            if (worker.joinable()) {
                worker.join();
            }
        } catch (...) {
        }
    }

    void rethrow_if_failed() {
        std::exception_ptr failure_copy;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            failure_copy = failure;
        }

        if (failure_copy) {
            std::rethrow_exception(failure_copy);
        }
    }

    void loop() noexcept {
        try {
            while (!stop_requested.load(std::memory_order_acquire)) {
                std::vector<Individual> exported;
                std::size_t generation_after_step = 0;
                bool done = false;

                {
                    std::lock_guard<std::mutex> lock(state_mutex);

                    if (algorithm->done()) {
                        done = true;
                    } else {
                        algorithm->step();
                        generation_after_step = algorithm->generation();

                        const std::size_t interval = spec.migration.interval_generations;
                        if (interval != 0 && generation_after_step % interval == 0) {
                            exported = algorithm->export_migrants(spec.migration.export_count);
                        }

                        done = algorithm->done();
                    }
                }

                if (!exported.empty() && !migration_channel.closed()) {
                    migration::Batch batch;
                    batch.source_island = spec.island_id;
                    batch.migrants.reserve(exported.size());

                    for (Individual& individual : exported) {
                        batch.migrants.push_back(migration::Migrant{
                            spec.island_id,
                            std::move(individual),
                            static_cast<std::uint64_t>(generation_after_step),
                        });
                    }

                    migration_channel.publish(std::move(batch));
                }

                if (spec.migration.import_count != 0 && !migration_channel.closed()) {
                    std::vector<migration::Migrant> imported =
                        migration_channel.try_take(spec.island_id, spec.migration.import_count);

                    if (!imported.empty()) {
                        std::vector<Individual> migrants;
                        migrants.reserve(imported.size());

                        for (migration::Migrant& migrant : imported) {
                            migrants.push_back(std::move(migrant.individual));
                        }

                        std::lock_guard<std::mutex> lock(state_mutex);
                        algorithm->import_migrants(std::move(migrants));
                    }
                }

                if (done) {
                    break;
                }

                // эта штука сейчас нужна только затем, чтобы при выключенной миграции остров не
                // крутил пустой цикл слишком агрессивно.
                // todo: когда будет нормальный, не stub алгоритм, эту штуку можно будет убрать
                if (spec.migration.interval_generations == 0) {
                    std::this_thread::sleep_for(k_idle_backoff);
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                failure = std::current_exception();
            }
            failed.store(true, std::memory_order_release);
        }

        running.store(false, std::memory_order_release);
        finished.store(true, std::memory_order_release);
    }

    IslandSpec spec;
    std::unique_ptr<Algorithm> algorithm;
    migration::Channel& migration_channel;

    mutable std::mutex state_mutex;
    std::thread worker;

    BitMatrix materialized_window;
    std::vector<std::size_t> owned_rows;

    bool initialized = false;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};
    std::atomic<bool> finished{false};

    std::exception_ptr failure;
    std::atomic<bool> failed{false};
};

Island::Island(IslandSpec spec,
               std::unique_ptr<Algorithm> algorithm,
               migration::Channel& migration_channel)
    : impl_(std::make_unique<Impl>(std::move(spec), std::move(algorithm), migration_channel)) {}

Island::~Island() = default;

Island::Island(Island&&) noexcept = default;
Island& Island::operator=(Island&&) noexcept = default;

void Island::initialize(const RowWindow& window) { impl_->initialize(window); }

void Island::start() { impl_->start(); }

void Island::request_stop() noexcept { impl_->request_stop(); }

void Island::wait() { impl_->wait(); }

bool Island::running() const noexcept { return impl_->is_running(); }

bool Island::stop_requested() const noexcept { return impl_->is_stop_requested(); }

bool Island::finished() const noexcept { return impl_->is_finished(); }

std::size_t Island::island_id() const noexcept { return impl_->island_id(); }

Individual Island::best_individual() const { return impl_->best_individual(); }

RunStats Island::stats() const { return impl_->stats(); }

IslandSnapshot Island::snapshot() const { return impl_->snapshot(); }

const char* Island::name() const noexcept { return impl_->name(); }

} // namespace bmmpy::ga
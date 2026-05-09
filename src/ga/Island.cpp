#include "bmmpy/ga/island.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace bmmpy::ga {
namespace {
constexpr auto k_idle_backoff = std::chrono::milliseconds(1);
};

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
        wait();
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

        initialized = true;
        stop_requested.store(false, std::memory_order_release);
        finished.store(false, std::memory_order_release);
        running.store(false, std::memory_order_release);
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
            algorithm->best_individual(),
        };
    }

    const char* name() const noexcept { return algorithm->name(); }

    std::size_t island_id() const noexcept { return spec.island_id; }

private:
    void loop() noexcept {
        // todo
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
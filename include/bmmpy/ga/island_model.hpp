#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/algorithm.hpp"
#include "bmmpy/ga/migration/channel.hpp"
#include "bmmpy/ga/types.hpp"

#include <memory>
#include <vector>

namespace bmmpy::ga {
struct IslandModelConfig {
    std::vector<IslandSpec> islands;
};

class IslandModel final {
public:
    IslandModel(IslandModelConfig config,
                AlgorithmFactory algorithm_factory,
                std::unique_ptr<migration::Channel> migration_channel);
    ~IslandModel();

    IslandModel(IslandModel&&) noexcept;
    IslandModel& operator=(IslandModel&&) noexcept;

    IslandModel(const IslandModel&) = delete;
    IslandModel& operator=(const IslandModel&) = delete;

    void initialize(const ::bmmpy::RowWindow& window);

    // start() -> working... -> request_stop() -> wait() means: "start the optimization, and when
    // you can, stop it as soon as possible, and wait until it's fully stopped start() -> wait()
    // means: "wait until the optimization is fully done, then return" request_stop() without wait()
    // means: "initialize the stopping process, but don't block immediately, you can check the
    // status with running() and stop_requested()"
    void start();
    void request_stop() noexcept; // please, begin stopping the optimization asap
    void wait();                  // lock until system is fully stopped

    bool running() const noexcept;
    bool stop_requested() const noexcept;

    Individual best_individual() const;
    IslandModelSnapshot snapshot() const;

    Individual run_to_completion(const ::bmmpy::RowWindow& window);

    const char* name() const noexcept { return "island_model"; }

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace bmmpy::ga
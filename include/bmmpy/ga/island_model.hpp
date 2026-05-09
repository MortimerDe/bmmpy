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

    void start();
    void request_stop() noexcept;
    void wait();

    bool running() const noexcept;
    bool stop_requested() const noexcept;

    Individual best_individual() const;
    IslandModelSnapshot snapshot() const;

    Individual optimize(const ::bmmpy::RowWindow& window);

    const char* name() const noexcept { return "island_model"; }

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace bmmpy::ga
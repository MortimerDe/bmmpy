#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/algorithm.hpp"
#include "bmmpy/ga/migration_channel.hpp"
#include "bmmpy/ga/types.hpp"

#include <memory>

namespace bmmpy::ga {
struct IslandModelConfig {
    std::size_t island_count = 8; // todo: or 0 for auto-detect?
    MigrationPolicy migration;
};

class IslandModel final {
    IslandModel(IslandModelConfig config,
                AlgorithmFactory algorithm_factory,
                std::unique_ptr<MigrationChannel> migration_channel);
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

    Individual run(const ::bmmpy::RowWindow& window);

    const char* name() const noexcept { return "island_model"; }

private:
    class Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace bmmpy::ga
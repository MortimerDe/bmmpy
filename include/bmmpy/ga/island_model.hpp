#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/types.hpp"
#include "bmmpy/ga/worker.hpp"

#include <memory>
#include <vector>

namespace bmmpy::ga {

struct IslandModelConfig {
    std::size_t island_count = 8;
    MigrationPolicy migration;
};

class IslandModel final {
public:
    IslandModel(IslandModelConfig config, std::unique_ptr<Worker> prototype);
    IslandModel(IslandModelConfig config, std::vector<std::unique_ptr<Worker>> workers);

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
    IslandModelConfig _config;
};

} // namespace bmmpy::ga
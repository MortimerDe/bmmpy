// types.hpp
#pragma once

#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace bmmpy::ga {

using Individual = std::vector<::bmmpy::Candidate>;
using Population = std::vector<Individual>;

struct StopCriteria {
    std::optional<std::size_t> max_generations;
    std::optional<std::size_t> max_stale_generations;
    std::optional<std::uint64_t> target_total_weight;
};

struct RunStats {
    std::size_t generations = 0;
    std::size_t evaluations = 0;
    std::size_t stale_generations = 0;
    std::uint64_t seed = 0;
};

// Local to each island.
struct MigrationPolicy {
    std::size_t interval_generations = 32;
    std::size_t export_count = 2;
    std::size_t import_count = 2;
    std::size_t shared_pool_capacity = 128;
};

struct IslandSpec {
    std::size_t island_id = 0;
    MigrationPolicy migration;
};

struct IslandSnapshot {
    std::size_t island_id = 0;
    bool running = false;
    bool stop_requested = false;
    bool finished = false;
    RunStats stats;
    Individual best_individual;
};

struct IslandModelSnapshot {
    bool running = false;
    bool stop_requested = false;
    std::size_t island_count = 0;
    std::uint64_t total_generations = 0;
    Individual best_individual;
    std::vector<IslandSnapshot> islands;
};

} // namespace bmmpy::ga
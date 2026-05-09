#pragma once

#include "bmmpy/core/row_window.hpp"
#include "bmmpy/ga/algorithm.hpp"
#include "bmmpy/ga/migration/channel.hpp"
#include "bmmpy/ga/types.hpp"

#include <memory>

namespace bmmpy::ga {

class Island final {
public:
    Island(IslandSpec spec,
           std::unique_ptr<Algorithm> algorithm,
           migration::Channel& migration_channel);
    ~Island();

    Island(Island&&) noexcept;
    Island& operator=(Island&&) noexcept;

    Island(const Island&) = delete;
    Island& operator=(const Island&) = delete;

    void initialize(const ::bmmpy::RowWindow& window);

    void run();
    void request_stop() noexcept;
    void wait();

    bool running() const noexcept;
    bool stop_requested() const noexcept;
    bool finished() const noexcept;

    std::size_t island_id() const noexcept;

    Individual best_individual() const;
    RunStats stats() const;
    IslandSnapshot snapshot() const;

    const char* algorithm_name() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace bmmpy::ga
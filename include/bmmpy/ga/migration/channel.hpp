#pragma once

#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy::ga::migration {

struct Migrant {
    std::size_t source_island = 0;
    Individual individual;
    std::uint64_t generation = 0;
};

struct Batch {
    std::size_t source_island = 0;
    std::vector<Migrant> migrants;
};

enum class OverflowPolicy {
    DropOldest,
    DropNewest,
    RejectPublish,
};

struct PublishResult {
    std::size_t accepted = 0;
    std::size_t dropped = 0;
    bool closed = false;
};

struct ChannelStats {
    std::size_t size = 0;
    std::size_t capacity = 0;
    std::uint64_t published = 0;
    std::uint64_t accepted = 0;
    std::uint64_t dropped = 0;
    std::uint64_t taken = 0;
    bool closed = false;
};

class Channel {
public:
    virtual ~Channel() = default;

    virtual PublishResult publish(Batch batch) = 0;

    virtual std::vector<Migrant> try_take(std::size_t consumer_island, std::size_t max_count) = 0;

    virtual void clear() = 0;
    virtual void reset() = 0;
    virtual void close() noexcept = 0;
    virtual bool closed() const noexcept = 0;

    virtual ChannelStats stats() const = 0;

    virtual const char* name() const noexcept = 0;
};

} // namespace bmmpy::ga::migration
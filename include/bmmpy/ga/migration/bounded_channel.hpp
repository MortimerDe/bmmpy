#pragma once

#include "bmmpy/ga/migration/channel.hpp"

#include <deque>
#include <mutex>

namespace bmmpy::ga::migration {

class BoundedChannel final : public Channel {
public:
    BoundedChannel(std::size_t capacity,
                   OverflowPolicy overflow_policy = OverflowPolicy::DropOldest);

    PublishResult publish(Batch batch) override;

    std::vector<Migrant> try_take(std::size_t consumer_island, std::size_t max_count) override;

    void clear() override;
    void close() noexcept override;
    bool closed() const noexcept override;

    ChannelStats stats() const override;

    const char* name() const noexcept override { return "bounded_channel"; }

private:
    struct Entry {
        std::size_t source_island = 0;
        Individual individual;
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
    };

    PublishResult publish_drop_oldest(Batch batch);
    PublishResult publish_drop_newest(Batch batch);
    PublishResult publish_reject(Batch batch);

    std::size_t capacity_ = 0;
    OverflowPolicy overflow_policy_ = OverflowPolicy::DropOldest;

    mutable std::mutex mutex_;
    std::deque<Entry> pool_;
    std::uint64_t next_sequence_ = 0;
    bool closed_ = false;

    std::uint64_t published_ = 0;
    std::uint64_t accepted_ = 0;
    std::uint64_t dropped_ = 0;
    std::uint64_t taken_ = 0;
};

} // namespace bmmpy::ga::migration
#include "bmmpy/ga/migration/bounded_channel.hpp"

#include "bmmpy/ga/migration/channel.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace bmmpy::ga::migration {

BoundedChannel::BoundedChannel(const std::size_t capacity, const OverflowPolicy overflow_policy)
    : capacity_(capacity), overflow_policy_(overflow_policy) {
    if (capacity_ == 0) {
        throw std::invalid_argument("BoundedChannel: capacity must be > 0");
    }
}

PublishResult BoundedChannel::publish(Batch batch) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (closed_) {
        return PublishResult{0, batch.migrants.size(), true};
    }

    published_ += batch.migrants.size();

    switch (overflow_policy_) {
    case OverflowPolicy::DropOldest:
        return publish_drop_oldest(std::move(batch));
    case OverflowPolicy::DropNewest:
        return publish_drop_newest(std::move(batch));
    case OverflowPolicy::RejectPublish:
        return publish_reject(std::move(batch));
    }

    return {};
}

PublishResult BoundedChannel::publish_drop_oldest(Batch batch) {
    PublishResult result;

    for (Migrant& migrant : batch.migrants) {
        while (pool_.size() >= capacity_) {
            pool_.pop_front();
            ++dropped_;
            ++result.dropped;
        }

        pool_.push_back(Entry{
            batch.source_island,
            std::move(migrant.individual),
            migrant.generation,
            next_sequence_++,
        });

        ++accepted_;
        ++result.accepted;
    }

    return result;
}

PublishResult BoundedChannel::publish_drop_newest(Batch batch) {
    PublishResult result;

    for (Migrant& migrant : batch.migrants) {
        if (pool_.size() >= capacity_) {
            ++dropped_;
            ++result.dropped;
            continue;
        }

        pool_.push_back(Entry{
            batch.source_island,
            std::move(migrant.individual),
            migrant.generation,
            next_sequence_++,
        });

        ++accepted_;
        ++result.accepted;
    }

    return result;
}

PublishResult BoundedChannel::publish_reject(Batch batch) {
    PublishResult result;

    if (pool_.size() + batch.migrants.size() > capacity_) {
        result.dropped = batch.migrants.size();
        dropped_ += batch.migrants.size();
        return result;
    }

    for (Migrant& migrant : batch.migrants) {
        pool_.push_back(Entry{
            batch.source_island,
            std::move(migrant.individual),
            migrant.generation,
            next_sequence_++,
        });

        ++accepted_;
        ++result.accepted;
    }

    return result;
}

std::vector<Migrant> BoundedChannel::try_take(const std::size_t consumer_island,
                                              const std::size_t max_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Migrant> out;
    out.reserve(max_count);

    if (max_count == 0 || pool_.empty()) {
        return out;
    }

    for (auto it = pool_.begin(); it != pool_.end() && out.size() < max_count;) {
        if (it->source_island == consumer_island) { // incest alert ⚠️⚠️⚠️
            ++it;
            continue;
        }

        out.push_back(Migrant{
            it->source_island,
            std::move(it->individual),
            it->generation,
        });

        it = pool_.erase(it);
        ++taken_;
    }

    return out;
}

void BoundedChannel::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.clear();
}

void BoundedChannel::close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
}

bool BoundedChannel::closed() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

ChannelStats BoundedChannel::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ChannelStats{
        pool_.size(),
        capacity_,
        published_,
        accepted_,
        dropped_,
        taken_,
        closed_,
    };
}

} // namespace bmmpy::ga::migration
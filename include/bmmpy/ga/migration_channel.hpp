#pragma once

#include "bmmpy/ga/types.hpp"

#include <cstddef>
#include <vector>

namespace bmmpy::ga {
struct MigrantBatch {
    std::size_t source_island = 0;
    std::vector<Individual> individuals;
};

class MigrationChannel {
public:
    virtual ~MigrationChannel() = default;

    virtual void publish(MigrantBatch batch) = 0;

    virtual std::vector<Individual> try_take(std::size_t consumer_island,
                                             std::size_t max_count) = 0;
    virtual void clear() = 0;

    virtual const char* name() const noexcept = 0;
};
} // namespace bmmpy::ga
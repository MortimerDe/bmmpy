#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy {

struct ApplyResult {
    std::size_t applied_count = 0;
    std::uint64_t weight_improvement = 0;
};

} // namespace bmmpy
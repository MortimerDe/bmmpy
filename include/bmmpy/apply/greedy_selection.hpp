#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/types/candidate.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy {

struct ApplyResult {
    std::size_t applied_count = 0;
    std::uint64_t weight_improvement = 0;
};

class GreedySelection {
public:
    GreedySelection(std::uint64_t min_gain,
                    bool stochastic = false,
                    std::uint64_t seed = 0) noexcept
        : _min_gain(min_gain), _stochastic(stochastic),
          _prng_state(seed == 0 ? 0xDEADBull : seed) {}

    ApplyResult apply(BitMatrix& matrix,
                      const std::vector<std::size_t>& window_rows,
                      const std::vector<Candidate>& candidates);

private:
    std::uint64_t next_random() noexcept {
        std::uint64_t x = _prng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        _prng_state = x;
        return x;
    }

    std::uint64_t _min_gain = 0;
    bool _stochastic = false;
    std::uint64_t _prng_state = 0xDEADB0BAull;
};

} // namespace bmmpy
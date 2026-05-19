#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy::detail {

constexpr std::uint64_t k_default_xorshift64_seed = 0x9E3779B97F4A7C15ull;

class XorShift64 {
public:
    explicit XorShift64(std::uint64_t seed = 0) noexcept { reseed(seed); }

    void reseed(std::uint64_t seed) noexcept {
        _state = seed == 0 ? k_default_xorshift64_seed : seed;
    }

    std::uint64_t next_u64() noexcept {
        std::uint64_t x = _state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        _state = x;
        return x;
    }

    std::size_t next_index(std::size_t upper_bound) noexcept {
        if (upper_bound == 0)
            return 0;
        return static_cast<std::size_t>(next_u64() % upper_bound);
    }

    double next_unit_double() noexcept {
        constexpr double k_inv_53 = 1.0 / 9007199254740992.0;
        return static_cast<double>(next_u64() >> 11) * k_inv_53;
    }

private:
    std::uint64_t _state = k_default_xorshift64_seed;
};

} // namespace bmmpy::detail
#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bmmpy::detail {

inline std::uint32_t popcount64(std::uint64_t value) noexcept {
#if defined(_MSC_VER)
    return static_cast<std::uint32_t>(__popcnt64(value));
#else
    return static_cast<std::uint32_t>(__builtin_popcountll(value));
#endif
}

inline unsigned ctz64(std::uint64_t value) noexcept {
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_ctzll(value));
#endif
}

inline bool parity64(std::uint64_t value) noexcept {
    return (popcount64(value) & 1u) != 0;
}

} // namespace bmmpy::detail
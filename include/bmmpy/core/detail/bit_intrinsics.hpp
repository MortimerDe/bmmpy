#pragma once

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bmmpy::detail {

inline std::uint32_t popcount64(std::uint64_t value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint32_t>(__builtin_popcountll(value));
#else
    value -= (value >> 1) & 0x5555555555555555ull;
    value = (value & 0x3333333333333333ull) + ((value >> 2) & 0x3333333333333333ull);
    value = (value + (value >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return static_cast<std::uint32_t>((value * 0x0101010101010101ull) >> 56);
#endif
}

inline unsigned ctz64(std::uint64_t value) noexcept {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<unsigned>(index);
#elif defined(_MSC_VER)
    unsigned long index = 0;
    const auto low = static_cast<unsigned long>(value & 0xffffffffull);
    if (low != 0) {
        _BitScanForward(&index, low);
        return static_cast<unsigned>(index);
    }

    _BitScanForward(&index, static_cast<unsigned long>(value >> 32));
    return static_cast<unsigned>(index + 32);
#else
    return static_cast<unsigned>(__builtin_ctzll(value));
#endif
}

inline bool parity64(std::uint64_t value) noexcept { return (popcount64(value) & 1u) != 0; }

inline bool runtime_supports_avx2() noexcept {
#if !defined(BMMPY_HAS_AVX2_BACKEND)
    return false;
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {0, 0, 0, 0};

    __cpuid(regs, 0);
    if (regs[0] < 7)
        return false;

    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!osxsave || !avx)
        return false;

    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6u) != 0x6u)
        return false;

    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) &&                                                 \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

} // namespace bmmpy::detail
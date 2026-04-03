#include "bmmpy/core/bit_ops.hpp"

#include <immintrin.h>

namespace bmmpy::detail {

void row_xor_avx2(std::uint64_t* dst,
                  const std::uint64_t* src,
                  std::size_t len) noexcept {
    std::size_t i = 0;

    for (; i + 4 <= len; i += 4) {
        const __m256i lhs =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(dst + i));
        const __m256i rhs =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));
        const __m256i out = _mm256_xor_si256(lhs, rhs);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst + i), out);
    }

    for (; i < len; ++i)
        dst[i] ^= src[i];
}

void row_swap_avx2(std::uint64_t* a,
                   std::uint64_t* b,
                   std::size_t len) noexcept {
    std::size_t i = 0;

    for (; i + 4 <= len; i += 4) {
        const __m256i va =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(b + i));

        _mm256_store_si256(reinterpret_cast<__m256i*>(a + i), vb);
        _mm256_store_si256(reinterpret_cast<__m256i*>(b + i), va);
    }

    for (; i < len; ++i) {
        const std::uint64_t tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

} // namespace bmmpy::detail
#include "bmmpy/core/detail/bit_ops.hpp"

#include <immintrin.h>

namespace bmmpy::detail {
namespace {

inline std::uint64_t hsum_u64x4(__m256i value) noexcept {
    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

} // namespace

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

std::uint64_t row_popcount_avx2(const std::uint64_t* src,
                                std::size_t len) noexcept {
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    const __m256i lookup = _mm256_setr_epi8(0,
                                            1,
                                            1,
                                            2,
                                            1,
                                            2,
                                            2,
                                            3,
                                            1,
                                            2,
                                            2,
                                            3,
                                            2,
                                            3,
                                            3,
                                            4,
                                            0,
                                            1,
                                            1,
                                            2,
                                            1,
                                            2,
                                            2,
                                            3,
                                            1,
                                            2,
                                            2,
                                            3,
                                            2,
                                            3,
                                            3,
                                            4);
    const __m256i zero = _mm256_setzero_si256();

    std::size_t i = 0;
    __m256i total = _mm256_setzero_si256();

    for (; i + 4 <= len; i += 4) {
        const __m256i v =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(src + i));

        const __m256i lo = _mm256_and_si256(v, low_mask);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);

        const __m256i popcnt_bytes = _mm256_add_epi8(
            _mm256_shuffle_epi8(lookup, lo), _mm256_shuffle_epi8(lookup, hi));

        total = _mm256_add_epi64(total, _mm256_sad_epu8(popcnt_bytes, zero));
    }

    std::uint64_t result = hsum_u64x4(total);

    for (; i < len; ++i)
        result += static_cast<std::uint64_t>(__builtin_popcountll(src[i]));

    return result;
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
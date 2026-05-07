#include "bmmpy/math/detail/fwht_ops.hpp"

#include <immintrin.h>

namespace bmmpy::detail {

void comb_i16_avx2(std::int16_t* left, std::int16_t* right, std::size_t len) noexcept {
    constexpr std::size_t k_lanes = 16;

    std::size_t i = 0;
    for (; i + k_lanes <= len; i += k_lanes) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + i));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(left + i), _mm256_add_epi16(a, b));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(right + i), _mm256_sub_epi16(a, b));
    }

    comb_i16_scalar(left + i, right + i, len - i);
}

void comb_i32_avx2(std::int32_t* left, std::int32_t* right, std::size_t len) noexcept {
    constexpr std::size_t k_lanes = 8;

    std::size_t i = 0;
    for (; i + k_lanes <= len; i += k_lanes) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + i));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(left + i), _mm256_add_epi32(a, b));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(right + i), _mm256_sub_epi32(a, b));
    }

    comb_i32_scalar(left + i, right + i, len - i);
}

} // namespace bmmpy::detail
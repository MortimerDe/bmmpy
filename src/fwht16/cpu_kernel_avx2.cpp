#include "bmmpy/fwht16/cpu_kernel.hpp"

#include <immintrin.h>

namespace bmmpy::fwht16 {
namespace {

void fwht16_comb_avx2(std::int16_t* left, std::int16_t* right, std::size_t len) noexcept {
    constexpr std::size_t k_lanes = 16;

    std::size_t i = 0;
    for (; i + k_lanes <= len; i += k_lanes) {
        const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + i));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(left + i), _mm256_add_epi16(a, b));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(right + i), _mm256_sub_epi16(a, b));
    }

    // scalar tail
    for (; i < len; ++i) {
        const std::int16_t u = left[i];
        const std::int16_t v = right[i];
        left[i] = static_cast<std::int16_t>(u + v);
        right[i] = static_cast<std::int16_t>(u - v);
    }
}

} // namespace

void fwht16_avx2(std::int16_t* data) noexcept {
    for (std::size_t len = 1; len < Fwht16Constants::k_spectrum_size; len <<= 1) {
        const std::size_t step = len << 1;

        for (std::size_t base = 0; base < Fwht16Constants::k_spectrum_size; base += step) {
            std::int16_t* left = data + base;
            std::int16_t* right = left + len;
            fwht16_comb_avx2(left, right, len);
        }
    }
}

} // namespace bmmpy::fwht16
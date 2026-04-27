#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"

namespace bmmpy::detail {

void row_xor_scalar(std::uint64_t* dst, const std::uint64_t* src, std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i)
        dst[i] ^= src[i];
}

std::uint64_t row_popcount_scalar(const std::uint64_t* src, std::size_t len) noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < len; ++i)
        total += static_cast<std::uint64_t>(popcount64(src[i]));
    return total;
}

std::uint64_t row_and_popcount_scalar(const std::uint64_t* lhs,
                                      const std::uint64_t* rhs,
                                      std::size_t len) noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < len; ++i)
        total += static_cast<std::uint64_t>(popcount64(lhs[i] & rhs[i]));
    return total;
}

void row_swap_scalar(std::uint64_t* a, std::uint64_t* b, std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint64_t tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

} // namespace bmmpy::detail
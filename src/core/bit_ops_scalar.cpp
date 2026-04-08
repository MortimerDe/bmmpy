#include "bmmpy/core/detail/bit_ops.hpp"

namespace bmmpy::detail {

void row_xor_scalar(std::uint64_t* dst,
                    const std::uint64_t* src,
                    std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i)
        dst[i] ^= src[i];
}

std::uint64_t row_popcount_scalar(const std::uint64_t* src,
                                  std::size_t len) noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < len; ++i)
        total += static_cast<std::uint64_t>(__builtin_popcountll(src[i]));
    return total;
}

void row_swap_scalar(std::uint64_t* a,
                     std::uint64_t* b,
                     std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint64_t tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;
    }
}

} // namespace bmmpy::detail
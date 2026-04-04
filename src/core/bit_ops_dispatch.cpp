#include "bmmpy/core/detail/bit_ops.hpp"

namespace bmmpy::detail {
namespace {

bool cpu_supports_avx2() noexcept {
#if defined(BMMPY_HAS_AVX2_BACKEND) &&                                         \
    (defined(__GNUC__) || defined(__clang__)) &&                               \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||            \
     defined(_M_IX86))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

} // namespace

const BitOps& bit_ops() noexcept {
    static const BitOps ops = [] {
        BitOps selected;
        selected.row_xor = &row_xor_scalar;
        selected.row_popcount = &row_popcount_scalar;
        selected.row_swap = &row_swap_scalar;

#if defined(BMMPY_HAS_AVX2_BACKEND)
        if (cpu_supports_avx2()) {
            selected.row_xor = &row_xor_avx2;
            selected.row_swap = &row_swap_avx2;
        }
#endif
        return selected;
    }();

    return ops;
}

} // namespace bmmpy::detail
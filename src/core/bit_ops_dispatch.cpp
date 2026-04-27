#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"

namespace bmmpy::detail {

const BitOps& bit_ops() noexcept {
    static const BitOps ops = [] {
        BitOps selected;
        selected.row_xor = &row_xor_scalar;
        selected.row_popcount = &row_popcount_scalar;
        selected.row_and_popcount = &row_and_popcount_scalar;
        selected.row_swap = &row_swap_scalar;

#if defined(BMMPY_HAS_AVX2_BACKEND)
        if (runtime_supports_avx2()) {
            selected.row_xor = &row_xor_avx2;
            selected.row_swap = &row_swap_avx2;
            selected.row_popcount = &row_popcount_avx2;
            selected.row_and_popcount = &row_and_popcount_avx2;
        }
#endif
        return selected;
    }();

    return ops;
}

} // namespace bmmpy::detail
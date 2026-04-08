#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/math/detail/fwht_ops.hpp"

namespace bmmpy::detail {

const FwhtOps& fwht_ops() noexcept {
    static const FwhtOps ops = [] {
        FwhtOps selected;
        selected.comb_i16 = &comb_i16_scalar;
        selected.comb_i32 = &comb_i32_scalar;

#if defined(BMMPY_HAS_AVX2_BACKEND)
        if (runtime_supports_avx2()) {
            selected.comb_i16 = &comb_i16_avx2;
            selected.comb_i32 = &comb_i32_avx2;
        }
#endif

        return selected;
    }();

    return ops;
}

} // namespace bmmpy::detail
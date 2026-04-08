#include "bmmpy/math/detail/fwht_ops.hpp"

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

const FwhtOps& fwht_ops() noexcept {
    static const FwhtOps ops = [] {
        FwhtOps selected;
        selected.comb_i16 = &comb_i16_scalar;
        selected.comb_i32 = &comb_i32_scalar;

#if defined(BMMPY_HAS_AVX2_BACKEND)
        if (cpu_supports_avx2()) {
            selected.comb_i16 = &comb_i16_avx2;
            selected.comb_i32 = &comb_i32_avx2;
        }
#endif

        return selected;
    }();

    return ops;
}

} // namespace bmmpy::detail
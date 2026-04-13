#include "bmmpy/fwht16/cpu_dispatch.hpp"

#include "bmmpy/core/detail/bit_intrinsics.hpp"

#include <stdexcept>

namespace bmmpy::fwht16 {
namespace {

Fwht16CpuDispatch make_scalar_dispatch() noexcept {
    return Fwht16CpuDispatch{
        Fwht16CpuBackend::scalar,
        &fwht16_scalar,
    };
}

} // namespace

Fwht16CpuDispatch resolve_cpu_dispatch(Fwht16CpuBackend requested) {
    switch (requested) {
    case Fwht16CpuBackend::auto_select:
        return make_scalar_dispatch();

    case Fwht16CpuBackend::scalar:
        return make_scalar_dispatch();

    case Fwht16CpuBackend::avx2:
#if defined(BMMPY_HAS_AVX2_BACKEND)
        if (bmmpy::detail::runtime_supports_avx2()) {
            throw std::runtime_error(
                "Fwht16 AVX2 dispatch is wired, but AVX2 kernel is not implemented");
        }
#endif
        throw std::runtime_error("Fwht16 AVX2 backend is not available");

    case Fwht16CpuBackend::avx512:
        throw std::runtime_error("Fwht16 AVX512 backend is not available");
    }

    throw std::invalid_argument("Fwht16: unknown CPU backend");
}

} // namespace bmmpy::fwht16
#pragma once

#include "bmmpy/fwht16/cpu_kernel.hpp"
#include "bmmpy/fwht16/types.hpp"

namespace bmmpy::fwht16 {

struct Fwht16CpuDispatch {
    Fwht16CpuBackend backend = Fwht16CpuBackend::scalar;
    Fwht16KernelFn kernel = nullptr;
};

Fwht16CpuDispatch resolve_cpu_dispatch(Fwht16CpuBackend requested);

} // namespace bmmpy::fwht16
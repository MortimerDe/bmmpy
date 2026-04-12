#include "bmmpy/fwht16/gpu_executor.hpp"

#include <stdexcept>

namespace bmmpy::fwht16 {
bool GpuFwht16Executor::available() noexcept { return false; }

Fwht16BatchResponse GpuFwht16Executor::run(const Fwht16BatchRequest&) const {
    throw std::runtime_error("GpuFwht16Executor: GPU backend is not implemented");
}
} // namespace bmmpy::fwht16
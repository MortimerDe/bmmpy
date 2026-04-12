#include "bmmpy/fwht16/cpu_executor.hpp"

#include <stdexcept>

namespace bmmpy::fwht16 {

namespace {
Fwht16CpuBackend resolve_cpu_backend(Fwht16CpuBackend requested) {
    switch (requested) {
    case Fwht16CpuBackend::auto_select:
        return Fwht16CpuBackend::scalar; // Only scalar is implemented for now.

    case Fwht16CpuBackend::scalar:
        return Fwht16CpuBackend::scalar;

    case Fwht16CpuBackend::avx2:
        throw std::runtime_error("CpuFwht16Executor: AVX2 backend is not implemented");

    case Fwht16CpuBackend::avx512:
        throw std::runtime_error("CpuFwht16Executor: AVX512 backend is not implemented");
    }
    throw std::runtime_error("CpuFwht16Executor: unknown CPU backend");
}
} // namespace

Fwht16BatchResponse CpuFwht16Executor::run(const Fwht16BatchRequest& request) const {
    Fwht16BatchResponse response;
    response.actual_backend = Fwht16Backend::cpu;
    response.actual_cpu_backend = resolve_cpu_backend(request.cpu_backend);
    response.mode = request.mode;
    response.batch_size = request.batch_size;

    if (request.mode == Fwht16ResultMode::topk) {
        response.topk_results.resize(request.batch_size * request.topk);
    } else {
        response.spectra.resize(request.batch_size * Fwht16Constants::k_spectrum_size, 0);
    }

    return response;
}

} // namespace bmmpy::fwht16
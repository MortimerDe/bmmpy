#include "bmmpy/fwht16/cpu_executor.hpp"

#include "bmmpy/fwht16/cpu_kernel.hpp"

#include <stdexcept>
#include <vector>

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

    std::vector<std::int16_t> scratch; // Reusable scratch space for FWHT computation.
    if (request.mode == Fwht16ResultMode::topk)
        scratch.resize(Fwht16Constants::k_spectrum_size);

    for (std::size_t sample_index = 0; sample_index < request.batch_size; ++sample_index) {
        std::int16_t* spectrum =
            request.mode == Fwht16ResultMode::spectrum
                ? response.spectra.data() + sample_index * Fwht16Constants::k_spectrum_size
                : scratch.data();

        build_histogram_16(request.samples[sample_index], spectrum);

        switch (response.actual_cpu_backend) {
        case Fwht16CpuBackend::auto_select:
        case Fwht16CpuBackend::scalar:
            fwht16_scalar(spectrum);
            break;
        case Fwht16CpuBackend::avx2:
        case Fwht16CpuBackend::avx512:
            throw std::runtime_error("CpuFwht16Executor: unresolved CPU backend");
        }

        if (request.mode == Fwht16ResultMode::topk) {
            extract_topk_16(
                spectrum, request.topk, response.topk_results.data() + sample_index * request.topk);
        }
    }

    return response;
}

} // namespace bmmpy::fwht16
#include "bmmpy/fwht16/cpu_executor.hpp"

#include "bmmpy/fwht16/cpu_dispatch.hpp"

#include <stdexcept>
#include <vector>

namespace bmmpy::fwht16 {

Fwht16BatchResponse CpuFwht16Executor::run(const Fwht16BatchRequest& request) const {
    Fwht16BatchResponse response;
    const Fwht16CpuDispatch dispatch = resolve_cpu_dispatch(request.cpu_backend);

    response.actual_backend = Fwht16Backend::cpu;
    response.actual_cpu_backend = dispatch.backend;
    response.mode = request.mode;
    response.batch_size = request.batch_size;

    if (request.mode == Fwht16ResultMode::topk) {
        response.topk_results.resize(request.batch_size * request.topk);
    } else {
        response.spectra.resize(request.batch_size * Fwht16Constants::k_spectrum_size, 0);
    }

    std::vector<std::int16_t> scratch;
    if (request.mode == Fwht16ResultMode::topk) {
        scratch.resize(Fwht16Constants::k_spectrum_size);
    }

    for (std::size_t sample_index = 0; sample_index < request.batch_size; ++sample_index) {
        std::int16_t* spectrum =
            request.mode == Fwht16ResultMode::spectrum
                ? response.spectra.data() + sample_index * Fwht16Constants::k_spectrum_size
                : scratch.data();

        build_histogram_16(request.samples[sample_index], spectrum);
        dispatch.kernel(spectrum);

        if (request.mode == Fwht16ResultMode::topk) {
            extract_topk_16(
                spectrum, request.topk, response.topk_results.data() + sample_index * request.topk);
        }
    }

    return response;
}

} // namespace bmmpy::fwht16
#include "bmmpy/fwht16/cpu_executor.hpp"

namespace bmmpy {

Fwht16BatchResponse CpuFwht16Executor::run(const Fwht16BatchRequest& request) const {
    Fwht16BatchResponse response;
    response.actual_backend = Fwht16Backend::cpu;
    response.mode = request.mode;
    response.batch_size = request.batch_size;

    if (request.mode == Fwht16ResultMode::topk) {
        response.topk_offsets.resize(request.batch_size + 1, 0);
    } else {
        response.spectra.resize(request.batch_size * Fwht16Constants::k_spectrum_size, 0);
    }

    return response;
}

} // namespace bmmpy
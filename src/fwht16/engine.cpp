#include "bmmpy/fwht16/engine.hpp"

#include "bmmpy/fwht16/cpu_executor.hpp"
#include "bmmpy/fwht16/gpu_executor.hpp"

#include <stdexcept>

namespace bmmpy::fwht16 {
namespace {

void validate_request(const Fwht16BatchRequest& request) {
    if (request.batch_size != 0 && request.samples == nullptr) {
        throw std::invalid_argument(
            "Fwht16Engine: samples must not be null when batch_size is non-zero");
    }

    if (request.mode == Fwht16ResultMode::topk && request.topk == 0) {
        throw std::invalid_argument("Fwht16Engine: topk must be non-zero in topk mode");
    }
}

} // namespace

Fwht16BatchResponse Fwht16Engine::run(const Fwht16BatchRequest& request) const {
    validate_request(request);

    switch (request.backend) {
    case Fwht16Backend::cpu:
        return CpuFwht16Executor{}.run(request);

    case Fwht16Backend::gpu:
        if (!GpuFwht16Executor::available()) {
            throw std::runtime_error("Fwht16Engine: GPU backend requested but not available");
        }
        return GpuFwht16Executor{}.run(request);

    case Fwht16Backend::auto_select:
        if (GpuFwht16Executor::available()) {
            return GpuFwht16Executor{}.run(request);
        }
        return CpuFwht16Executor{}.run(request);
    }

    throw std::invalid_argument("Fwht16Engine: unknown backend specified");
}
} // namespace bmmpy::fwht16
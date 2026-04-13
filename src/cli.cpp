#include "bmmpy/core/detail/bit_intrinsics.hpp"
#include "bmmpy/fwht16/constants.hpp"
#include "bmmpy/fwht16/engine.hpp"
#include "bmmpy/fwht16/types.hpp"
#include "bmmpy/math/fwht.hpp"
#include "bmmpy/stub.hpp"

#include <iostream>

int main() {
    std::cout << "Version: " << bmmpy::get_version() << std::endl;

    bmmpy::fwht16::ColumnMasks16 sample{};
    bmmpy::fwht16::Fwht16Engine engine;

    bmmpy::fwht16::Fwht16BatchRequest request;
    request.samples = &sample;
    request.batch_size = 1;
    request.backend = bmmpy::fwht16::Fwht16Backend::cpu;
    request.cpu_backend = bmmpy::fwht16::Fwht16CpuBackend::auto_select;
    request.mode = bmmpy::fwht16::Fwht16ResultMode::topk;
    request.topk = 4;

    const auto response = engine.run(request);

    std::cout << "Actual backend: "
              << (response.actual_cpu_backend == bmmpy::fwht16::Fwht16CpuBackend::scalar ? "scalar"
                                                                                         : "avx2")
              << std::endl;

    return 0;
}
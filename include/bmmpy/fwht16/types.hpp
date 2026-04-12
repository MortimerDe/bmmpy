#pragma once

#include "bmmpy/fwht16/constants.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bmmpy::fwht16 {
struct ColumnMasks16 {
    std::array<std::uint16_t, Fwht16Constants::k_cols> masks{};
};

enum class Fwht16Backend {
    auto_select,
    cpu,
    gpu,
};

enum class Fwht16CpuBackend {
    auto_select,
    scalar,
    avx2,
    avx512,
};

enum class Fwht16ResultMode {
    topk,
    spectrum,
};

struct Fwht16TopKItem {
    std::uint16_t mask = 0;
    std::uint16_t weight = 0;
};

struct Fwht16BatchRequest {
    const ColumnMasks16* samples = nullptr;
    std::size_t batch_size = 0;

    Fwht16Backend backend = Fwht16Backend::auto_select;
    Fwht16CpuBackend cpu_backend = Fwht16CpuBackend::auto_select;

    Fwht16ResultMode mode = Fwht16ResultMode::topk;
    std::size_t topk = 64;
};

struct Fwht16BatchResponse {
    Fwht16Backend actual_backend = Fwht16Backend::cpu;
    Fwht16CpuBackend actual_cpu_backend = Fwht16CpuBackend::scalar;
    Fwht16ResultMode mode = Fwht16ResultMode::topk;
    std::size_t batch_size = 0;

    // Flattened fixed-size top-k output.
    // Results for sample i occupy:
    // [i * topk, (i + 1) * topk)
    std::vector<Fwht16TopKItem> topk_results;

    // Flattened dense spectra.
    // Spectrum for sample i occupies:
    // [i * Fwht16Constants::k_spectrum_size,
    //  (i + 1) * Fwht16Constants::k_spectrum_size)
    std::vector<std::int16_t> spectra;
};

} // namespace bmmpy::fwht16
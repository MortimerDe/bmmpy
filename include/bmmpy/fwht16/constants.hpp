#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy {
struct Fwht16Constants {
    static constexpr std::size_t k_rows = 16;
    static constexpr std::size_t k_cols = 512;
    static constexpr std::size_t k_spectrum_size = std::size_t{1} << k_rows; // 2^16 = 65536
    static constexpr std::int16_t k_max_weight = 512;
};
} // namespace bmmpy
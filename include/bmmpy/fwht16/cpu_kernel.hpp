#pragma once

#include "bmmpy/fwht16/types.hpp"

#include <cstddef>
#include <cstdint>

namespace bmmpy::fwht16 {

using Fwht16KernelFn = void (*)(std::int16_t* data) noexcept;

void build_histogram_16(const ColumnMasks16& sample, std::int16_t* histogram) noexcept;

void fwht16_scalar(std::int16_t* data) noexcept;
void fwht16_avx2(std::int16_t* data) noexcept;

void extract_topk_16(const std::int16_t* spectrum, std::size_t k, Fwht16TopKItem* out) noexcept;
} // namespace bmmpy::fwht16
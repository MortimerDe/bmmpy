#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy::detail {

using CombI16Fn = void (*)(std::int16_t* left,
                           std::int16_t* right,
                           std::size_t len) noexcept;

using CombI32Fn = void (*)(std::int32_t* left,
                           std::int32_t* right,
                           std::size_t len) noexcept;

struct FwhtOps {
    CombI16Fn comb_i16 = nullptr;
    CombI32Fn comb_i32 = nullptr;
};

void comb_i16_scalar(std::int16_t* left,
                     std::int16_t* right,
                     std::size_t len) noexcept;

void comb_i32_scalar(std::int32_t* left,
                     std::int32_t* right,
                     std::size_t len) noexcept;

#if defined(BMMPY_HAS_AVX2_BACKEND)
void comb_i16_avx2(std::int16_t* left,
                   std::int16_t* right,
                   std::size_t len) noexcept;

void comb_i32_avx2(std::int32_t* left,
                   std::int32_t* right,
                   std::size_t len) noexcept;
#endif

const FwhtOps& fwht_ops() noexcept;

} // namespace bmmpy::detail
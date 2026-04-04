#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy::detail {

using RowXorFn = void (*)(std::uint64_t* dst,
                          const std::uint64_t* src,
                          std::size_t len) noexcept;

using RowPopcountFn = std::uint64_t (*)(const std::uint64_t* src,
                                        std::size_t len) noexcept;

using RowSwapFn = void (*)(std::uint64_t* a,
                           std::uint64_t* b,
                           std::size_t len) noexcept;

struct BitOps {
    RowXorFn row_xor = nullptr;
    RowPopcountFn row_popcount = nullptr;
    RowSwapFn row_swap = nullptr;
};

void row_xor_scalar(std::uint64_t* dst,
                    const std::uint64_t* src,
                    std::size_t len) noexcept;

std::uint64_t row_popcount_scalar(const std::uint64_t* src,
                                  std::size_t len) noexcept;

void row_swap_scalar(std::uint64_t* a,
                     std::uint64_t* b,
                     std::size_t len) noexcept;

#if defined(BMMPY_HAS_AVX2_BACKEND)
void row_xor_avx2(std::uint64_t* dst,
                  const std::uint64_t* src,
                  std::size_t len) noexcept;

std::uint64_t row_popcount_avx2(const std::uint64_t* src,
                                std::size_t len) noexcept;

void row_swap_avx2(std::uint64_t* a,
                   std::uint64_t* b,
                   std::size_t len) noexcept;
#endif

const BitOps& bit_ops() noexcept;

} // namespace bmmpy::detail
#pragma once

#include <cstdint>
#include <vector>

namespace bmmpy {
void fixed_weight_masks_u32(std::uint32_t n,
                            std::uint32_t k,
                            std::vector<std::uint32_t>& out);
void fixed_weight_masks_u64(std::uint32_t n,
                            std::uint32_t k,
                            std::vector<std::uint64_t>& out);
}
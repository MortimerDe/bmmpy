#pragma once

#include <cstddef>
#include <cstdint>

namespace bmmpy {

void fwht_inplace(std::int16_t* data, std::size_t size);
void fwht_inplace(std::int32_t* data, std::size_t size);

void calc_scores_and_order(const std::int16_t* h,
                           std::size_t m,
                           std::int32_t n,
                           std::int16_t* s_by_mask,
                           std::int32_t* order,
                           std::int32_t* cnt,
                           std::int32_t* off);

void calc_scores_and_order(const std::int32_t* h,
                           std::size_t m,
                           std::int32_t n,
                           std::int32_t* s_by_mask,
                           std::int32_t* order,
                           std::int32_t* cnt,
                           std::int32_t* off);

}
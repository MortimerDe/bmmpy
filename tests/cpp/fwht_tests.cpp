#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <vector>

using bmmpy::test::TestCase;

namespace {

void test_fixed_weight_masks_u32() {
    std::vector<std::uint32_t> masks;
    bmmpy::fixed_weight_masks_u32(5, 3, masks);

    bmmpy::test::require_eq<std::uint32_t>(masks,
                                           {
                                               0x07u,
                                               0x0Bu,
                                               0x0Du,
                                               0x0Eu,
                                               0x13u,
                                               0x15u,
                                               0x16u,
                                               0x19u,
                                               0x1Au,
                                               0x1Cu,
                                           },
                                           "fixed_weight_masks_u32");
}

void test_fixed_weight_masks_u64() {
    std::vector<std::uint64_t> masks;
    bmmpy::fixed_weight_masks_u64(6, 2, masks);

    bmmpy::test::require_eq<std::uint64_t>(masks,
                                           {
                                               0x03ull,
                                               0x05ull,
                                               0x06ull,
                                               0x09ull,
                                               0x0Aull,
                                               0x0Cull,
                                               0x11ull,
                                               0x12ull,
                                               0x14ull,
                                               0x18ull,
                                               0x21ull,
                                               0x22ull,
                                               0x24ull,
                                               0x28ull,
                                               0x30ull,
                                           },
                                           "fixed_weight_masks_u64");
}

void test_fwht_i32() {
    std::vector<std::int32_t> data = {1, 2, 3, 4};
    bmmpy::fwht_inplace(data.data(), data.size());
    bmmpy::test::require_eq<std::int32_t>(data, {10, -2, -4, 0}, "fwht_i32");
}

void test_fwht_i16_wrap() {
    std::vector<std::int16_t> data = {32767, 1};
    bmmpy::fwht_inplace(data.data(), data.size());

    bmmpy::test::require(data[0] == static_cast<std::int16_t>(-32768), "fwht_i16_wrap sum");
    bmmpy::test::require(data[1] == static_cast<std::int16_t>(32766), "fwht_i16_wrap diff");
}

void test_calc_scores_and_order_i32() {
    const std::vector<std::int32_t> h = {0, 0, 0, 4, 4};
    const std::int32_t n = 4;

    std::vector<std::int32_t> s_by_mask(h.size(), -1);
    std::vector<std::int32_t> order(h.size() - 1, -1);
    std::vector<std::int32_t> cnt(static_cast<std::size_t>(n) + 1, -1);
    std::vector<std::int32_t> off(static_cast<std::size_t>(n) + 1, -1);

    bmmpy::calc_scores_and_order(
        h.data(), h.size(), n, s_by_mask.data(), order.data(), cnt.data(), off.data());

    bmmpy::test::require_eq<std::int32_t>(s_by_mask, {0, 2, 2, 0, 0}, "scores_i32");
    bmmpy::test::require_eq<std::int32_t>(order, {3, 4, 1, 2}, "order_i32");
    bmmpy::test::require_eq<std::int32_t>(cnt, {2, 0, 2, 0, 0}, "cnt_i32");
}

void test_calc_scores_and_order_i16() {
    const std::vector<std::int16_t> h = {0, 4, 2, 0, -2};
    const std::int32_t n = 4;

    std::vector<std::int16_t> s_by_mask(h.size(), -1);
    std::vector<std::int32_t> order(h.size() - 1, -1);
    std::vector<std::int32_t> cnt(static_cast<std::size_t>(n) + 1, -1);
    std::vector<std::int32_t> off(static_cast<std::size_t>(n) + 1, -1);

    bmmpy::calc_scores_and_order(
        h.data(), h.size(), n, s_by_mask.data(), order.data(), cnt.data(), off.data());

    bmmpy::test::require_eq<std::int16_t>(s_by_mask, {0, 0, 1, 2, 3}, "scores_i16");
    bmmpy::test::require_eq<std::int32_t>(order, {1, 2, 3, 4}, "order_i16");
    bmmpy::test::require_eq<std::int32_t>(cnt, {1, 1, 1, 1, 0}, "cnt_i16");
}

void test_row_and_popcount_counts_intersection() {
    bmmpy::BitMatrix matrix(2, 130);

    for (std::size_t col : {0u, 63u, 64u, 100u, 129u})
        matrix.set(0, col, true);

    for (std::size_t col : {1u, 63u, 64u, 65u, 100u, 129u})
        matrix.set(1, col, true);

    const auto& ops = bmmpy::detail::bit_ops();
    const std::uint64_t expected = 4;

    bmmpy::test::require(ops.row_and_popcount != nullptr, "row_and_popcount dispatch is missing");
    bmmpy::test::require(ops.row_and_popcount(matrix.row_words(0),
                                              matrix.row_words(1),
                                              matrix.words_per_row()) == expected,
                         "row_and_popcount words_per_row mismatch");
    bmmpy::test::require(ops.row_and_popcount(matrix.row_words(1),
                                              matrix.row_words(0),
                                              matrix.words_per_row()) == expected,
                         "row_and_popcount symmetry mismatch");
    bmmpy::test::require(ops.row_and_popcount(matrix.row_words(0),
                                              matrix.row_words(1),
                                              matrix.stride_words()) == expected,
                         "row_and_popcount padded stride mismatch");
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"fixed_weight_masks_u32", &test_fixed_weight_masks_u32},
        {"fixed_weight_masks_u64", &test_fixed_weight_masks_u64},
        {"fwht_i32", &test_fwht_i32},
        {"fwht_i16_wrap", &test_fwht_i16_wrap},
        {"calc_scores_and_order_i32", &test_calc_scores_and_order_i32},
        {"calc_scores_and_order_i16", &test_calc_scores_and_order_i16},
        {"row_and_popcount_counts_intersection", &test_row_and_popcount_counts_intersection},
    };

    return bmmpy::test::run_tests(tests);
}
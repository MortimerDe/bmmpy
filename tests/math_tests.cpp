#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, std::string_view message) {
    if (!condition)
        fail(std::string(message));
}

template <typename T>
void require_eq(const std::vector<T>& actual,
                std::initializer_list<T> expected,
                std::string_view context) {
    require(actual.size() == expected.size(),
            std::string(context) + ": size mismatch");

    auto it = expected.begin();
    for (std::size_t i = 0; i < actual.size(); ++i, ++it) {
        if (actual[i] != *it)
            fail(std::string(context) + ": value mismatch");
    }
}

void test_fixed_weight_masks_u32() {
    std::vector<std::uint32_t> masks;
    bmmpy::fixed_weight_masks_u32(5, 3, masks);

    require_eq<std::uint32_t>(masks,
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

    require_eq<std::uint64_t>(masks,
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
    require_eq<std::int32_t>(data, {10, -2, -4, 0}, "fwht_i32");
}

void test_fwht_i16_wrap() {
    std::vector<std::int16_t> data = {32767, 1};
    bmmpy::fwht_inplace(data.data(), data.size());

    require(data[0] == static_cast<std::int16_t>(-32768), "fwht_i16_wrap sum");
    require(data[1] == static_cast<std::int16_t>(32766), "fwht_i16_wrap diff");
}

void test_calc_scores_and_order_i32() {
    const std::vector<std::int32_t> h = {0, 0, 0, 4, 4};
    const std::int32_t n = 4;

    std::vector<std::int32_t> s_by_mask(h.size(), -1);
    std::vector<std::int32_t> order(h.size() - 1, -1);
    std::vector<std::int32_t> cnt(static_cast<std::size_t>(n) + 1, -1);
    std::vector<std::int32_t> off(static_cast<std::size_t>(n) + 1, -1);

    bmmpy::calc_scores_and_order(h.data(),
                                 h.size(),
                                 n,
                                 s_by_mask.data(),
                                 order.data(),
                                 cnt.data(),
                                 off.data());

    require_eq<std::int32_t>(s_by_mask, {0, 2, 2, 0, 0}, "scores_i32");
    require_eq<std::int32_t>(order, {3, 4, 1, 2}, "order_i32");
    require_eq<std::int32_t>(cnt, {2, 0, 2, 0, 0}, "cnt_i32");
}

void test_calc_scores_and_order_i16() {
    const std::vector<std::int16_t> h = {0, 4, 2, 0, -2};
    const std::int32_t n = 4;

    std::vector<std::int16_t> s_by_mask(h.size(), -1);
    std::vector<std::int32_t> order(h.size() - 1, -1);
    std::vector<std::int32_t> cnt(static_cast<std::size_t>(n) + 1, -1);
    std::vector<std::int32_t> off(static_cast<std::size_t>(n) + 1, -1);

    bmmpy::calc_scores_and_order(h.data(),
                                 h.size(),
                                 n,
                                 s_by_mask.data(),
                                 order.data(),
                                 cnt.data(),
                                 off.data());

    require_eq<std::int16_t>(s_by_mask, {0, 0, 1, 2, 3}, "scores_i16");
    require_eq<std::int32_t>(order, {1, 2, 3, 4}, "order_i16");
    require_eq<std::int32_t>(cnt, {1, 1, 1, 1, 0}, "cnt_i16");
}

struct TestCase {
    const char* name;
    void (*fn)();
};

} // namespace

int main() {
    const TestCase tests[] = {
        {"fixed_weight_masks_u32", &test_fixed_weight_masks_u32},
        {"fixed_weight_masks_u64", &test_fixed_weight_masks_u64},
        {"fwht_i32", &test_fwht_i32},
        {"fwht_i16_wrap", &test_fwht_i16_wrap},
        {"calc_scores_and_order_i32", &test_calc_scores_and_order_i32},
        {"calc_scores_and_order_i16", &test_calc_scores_and_order_i16},
    };

    for (const TestCase& test : tests) {
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "All math tests passed\n";
    return 0;
}
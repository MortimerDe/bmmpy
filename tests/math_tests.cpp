#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/detail/bit_ops.hpp"
#include "bmmpy/math/comb.hpp"
#include "bmmpy/math/fwht.hpp"
#include "bmmpy/search/bruteforce_search.hpp"
#include "bmmpy/search/cuda_mitm_fwht_search.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/sa_selector.hpp"
#include "bmmpy/search/searcher.hpp"
#include "bmmpy/search/split_window_prep.hpp"
#include "bmmpy/stub.hpp"

#include <bmmpy/search/cuda_bruteforce_search.hpp>
#include <bmmpy/search/mitm_fwht_search.hpp>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, std::string_view message) {
    if (!condition)
        fail(std::string(message));
}

bmmpy::BitMatrix matrix_from_rows(std::initializer_list<std::string_view> rows) {
    const std::size_t row_count = rows.size();
    const std::size_t col_count = row_count == 0 ? 0 : rows.begin()->size();

    bmmpy::BitMatrix matrix(row_count, col_count);

    std::size_t row_index = 0;
    for (std::string_view row : rows) {
        require(row.size() == col_count, "row width mismatch in test fixture");

        for (std::size_t col_index = 0; col_index < col_count; ++col_index) {
            const char ch = row[col_index];
            if (ch == '1')
                matrix.set(row_index, col_index, true);
            else if (ch != '0')
                fail("test fixture must contain only '0' or '1'");
        }

        ++row_index;
    }

    return matrix;
}

bmmpy::BitMatrix make_sa_cluster_matrix() {
    return matrix_from_rows({
        "111111000000",
        "111111000000",
        "111111000000",
        "111111000000",
        "000000000000",
        "000000000000",
        "000000000000",
        "000000000000",
    });
}

bmmpy::BitMatrix make_cuda_equivalence_matrix(std::size_t rows = 28, std::size_t cols = 96) {
    bmmpy::BitMatrix matrix(rows, cols);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint64_t mix =
                (static_cast<std::uint64_t>(row + 1) * 0x9E3779B185EBCA87ull) ^
                (static_cast<std::uint64_t>(col + 3) * 0xC2B2AE3D27D4EB4Full) ^
                (static_cast<std::uint64_t>(row + col + 11) * 0x165667B19E3779F9ull);

            const std::uint64_t folded = mix ^ (mix >> 17) ^ (mix >> 33);
            if ((folded & 1ull) != 0)
                matrix.set(row, col, true);
        }
    }

    return matrix;
}

void require_same_candidates(const std::vector<bmmpy::Candidate>& actual,
                             const std::vector<bmmpy::Candidate>& expected,
                             std::string_view context) {
    require(actual.size() == expected.size(), std::string(context) + ": size mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].mask_u64() == expected[i].mask_u64(),
                std::string(context) + ": mask mismatch");
        require(actual[i].weight == expected[i].weight, std::string(context) + ": weight mismatch");
    }
}

template <typename T>
void require_eq(const std::vector<T>& actual,
                std::initializer_list<T> expected,
                std::string_view context) {
    require(actual.size() == expected.size(), std::string(context) + ": size mismatch");

    auto it = expected.begin();
    for (std::size_t i = 0; i < actual.size(); ++i, ++it) {
        if (actual[i] != *it)
            fail(std::string(context) + ": value mismatch");
    }
}

template <typename Fn> void expect_out_of_range(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::out_of_range, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::out_of_range");
}

template <typename Fn> void expect_invalid_argument(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::invalid_argument, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::invalid_argument");
}

template <typename Fn> void expect_runtime_error(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::runtime_error, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::runtime_error");
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

    bmmpy::calc_scores_and_order(
        h.data(), h.size(), n, s_by_mask.data(), order.data(), cnt.data(), off.data());

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

    bmmpy::calc_scores_and_order(
        h.data(), h.size(), n, s_by_mask.data(), order.data(), cnt.data(), off.data());

    require_eq<std::int16_t>(s_by_mask, {0, 0, 1, 2, 3}, "scores_i16");
    require_eq<std::int32_t>(order, {1, 2, 3, 4}, "order_i16");
    require_eq<std::int32_t>(cnt, {1, 1, 1, 1, 0}, "cnt_i16");
}

void test_row_and_popcount_counts_intersection() {
    bmmpy::BitMatrix matrix(2, 130);

    for (std::size_t col : {0u, 63u, 64u, 100u, 129u})
        matrix.set(0, col, true);

    for (std::size_t col : {1u, 63u, 64u, 65u, 100u, 129u})
        matrix.set(1, col, true);

    const auto& ops = bmmpy::detail::bit_ops();
    const std::uint64_t expected = 4;

    require(ops.row_and_popcount != nullptr, "row_and_popcount dispatch is missing");
    require(ops.row_and_popcount(
                matrix.row_words(0), matrix.row_words(1), matrix.words_per_row()) == expected,
            "row_and_popcount words_per_row mismatch");
    require(ops.row_and_popcount(
                matrix.row_words(1), matrix.row_words(0), matrix.words_per_row()) == expected,
            "row_and_popcount symmetry mismatch");
    require(ops.row_and_popcount(matrix.row_words(0), matrix.row_words(1), matrix.stride_words()) ==
                expected,
            "row_and_popcount padded stride mismatch");
}

void test_fwht_search_finds_best_candidate() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    bmmpy::FwhtSearch search;
    const auto window = matrix.row_window({0, 1});
    const auto candidates = search.search(window);

    require(candidates.size() == 3, "fwht_search candidate count");
    require(candidates[0].mask_u64() == 0x3ull, "fwht_search best mask");
    require(candidates[0].weight == 0, "fwht_search best weight");

    for (std::size_t i = 1; i < candidates.size(); ++i) {
        require(candidates[i - 1].weight <= candidates[i].weight, "fwht_search result order");
    }
}

void test_fwht_search_respects_k() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    bmmpy::FwhtSearch search({16, 1});
    const auto window = matrix.row_window({0, 1});
    const auto candidates = search.search(window);

    require(candidates.size() == 1, "fwht_search k limit");
    require(candidates[0].mask_u64() == 0x3ull, "fwht_search k best mask");
    require(candidates[0].weight == 0, "fwht_search k best weight");
}

void test_fwht_search_window_bounds() {
    bmmpy::BitMatrix matrix(1, 1);
    bmmpy::FwhtSearch search;

    expect_out_of_range([&] { (void)matrix.row_window({1}); }, "row_window bounds");
}

void test_searcher_interface_dispatch() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::FwhtSearch>(bmmpy::FwhtSearchConfig{16, 1});

    require(std::string_view(searcher->name()) == "fwht", "searcher name");

    const auto window = matrix.row_window({0, 1});
    const auto candidates = searcher->search(window);
    require(candidates.size() == 1, "searcher candidate count");
    require(candidates[0].mask_u64() == 0x3ull, "searcher best mask");
    require(candidates[0].weight == 0, "searcher best weight");
}

void test_mitm_fwht_matches_fwht_search() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "110010101001",
        "101100101100",
        "011010011001",
        "111000010111",
        "000111100011",
        "101011110000",
    });

    const std::vector<std::size_t> window_rows = {0, 1, 2, 3, 4, 5};

    bmmpy::FwhtSearch fwht({16, 8});
    bmmpy::MitmFwhtSearch mitm(bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 8});

    const auto window = matrix.row_window(window_rows);
    const auto expected = fwht.search(window);
    const auto actual = mitm.search(window);

    require_same_candidates(actual, expected, "mitm_fwht matches fwht");
}

void test_mitm_fwht_searcher_interface_dispatch() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::FwhtSearch baseline({16, 4});
    const auto window = matrix.row_window({0, 1, 2});
    const auto expected = baseline.search(window);

    std::unique_ptr<bmmpy::Searcher> searcher = std::make_unique<bmmpy::MitmFwhtSearch>(
        bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 4});

    require(std::string_view(searcher->name()) == "mitm_fwht", "mitm searcher name");

    const auto dispatch_window = matrix.row_window({0, 1, 2});
    const auto actual = searcher->search(dispatch_window);
    require_same_candidates(actual, expected, "mitm searcher dispatch");
}

void test_compact_split_window_collects_expected_patterns() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "11001",
        "00100",
        "01101",
        "01001",
    });

    const auto window = matrix.row_window({0, 1, 2, 3});
    const auto compact = bmmpy::build_compact_split_window(window, 2);

    require(compact.t == 4, "compact_split_window t");
    require(compact.low_bits == 2, "compact_split_window low_bits");
    require(compact.high_bits == 2, "compact_split_window high_bits");
    require(compact.total_weight == 4, "compact_split_window total_weight");

    require_eq<std::uint64_t>(compact.q,
                              {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}},
                              "compact_split_window q");
    require_eq<std::uint64_t>(compact.r,
                              {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{1}},
                              "compact_split_window r");
    require_eq<std::int32_t>(compact.multiplicity, {1, 1, 2}, "compact_split_window multiplicity");
}

void test_cuda_runtime_features_are_consistent() {
    const auto features = bmmpy::get_runtime_features();
    require(!(features.cuda_available && !features.cuda_compiled),
            "cuda_available implies cuda_compiled");
}

void test_cuda_mitm_fwht_searcher_interface_dispatch() {
    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::CudaMitmFwhtSearch>(bmmpy::CudaMitmFwhtSearchConfig{});

    require(std::string_view(searcher->name()) == "cuda_mitm_fwht", "cuda mitm searcher name");
}

void test_cuda_mitm_fwht_validates_window_size() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::CudaMitmFwhtSearch search;
    const auto window = matrix.row_window({0, 1, 2});

    expect_invalid_argument([&] { (void)search.search(window); }, "cuda mitm searcher window size");
}

void test_cuda_mitm_fwht_validates_low_bits() {
    bmmpy::BitMatrix matrix(28, 4);
    matrix.set(0, 0, true);

    bmmpy::CudaMitmFwhtSearch search({64, 28});
    const auto window = matrix.row_window({
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    });

    expect_invalid_argument([&] { (void)search.search(window); }, "cuda mitm searcher low_bits");
}

void test_cuda_mitm_fwht_rejects_non_int16_safe_total_weight() {
    bmmpy::BitMatrix matrix(28, 32768);
    for (std::size_t col = 0; col < 32768; ++col)
        matrix.set(0, col, true);

    bmmpy::CudaMitmFwhtSearch search;
    const auto window = matrix.row_window({
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    });

    expect_invalid_argument([&] { (void)search.search(window); },
                            "cuda mitm searcher int16-safe total_weight");
}

void test_cuda_mitm_fwht_matches_cpu_mitm_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix();

    std::vector<std::size_t> rows(28);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::MitmFwhtSearch cpu(bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 8});
    bmmpy::CudaMitmFwhtSearch gpu(bmmpy::CudaMitmFwhtSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    require_same_candidates(actual, expected, "cuda_mitm_fwht matches cpu mitm");
}

void test_cuda_mitm_fwht_supports_128_candidates_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix();

    std::vector<std::size_t> rows(28);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::MitmFwhtSearch cpu(bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 128});
    bmmpy::CudaMitmFwhtSearch gpu(bmmpy::CudaMitmFwhtSearchConfig{128, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    require_same_candidates(actual, expected, "cuda_mitm_fwht supports 128 candidates");
}

void test_bruteforce_search_respects_k() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    bmmpy::BruteforceSearch search({1, 0});
    const auto window = matrix.row_window({0, 1});
    const auto candidates = search.search(window);

    require(candidates.size() == 1, "bruteforce_search k limit");
    require(candidates[0].mask_u64() == 0x3ull, "bruteforce_search k best mask");
    require(candidates[0].weight == 0, "bruteforce_search k best weight");
}

void test_bruteforce_search_matches_fwht_search() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "110010101001",
        "101100101100",
        "011010011001",
        "111000010111",
        "000111100011",
        "101011110000",
    });

    const std::vector<std::size_t> window_rows = {0, 1, 2, 3, 4, 5};

    bmmpy::FwhtSearch fwht({16, 8});
    bmmpy::BruteforceSearch brute({8, 0});

    const auto window = matrix.row_window(window_rows);
    const auto expected = fwht.search(window);
    const auto actual = brute.search(window);

    require_same_candidates(actual, expected, "bruteforce matches fwht");
}

void test_bruteforce_searcher_interface_dispatch() {
    const bmmpy::BitMatrix matrix = matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::FwhtSearch baseline({16, 4});
    const auto window = matrix.row_window({0, 1, 2});
    const auto expected = baseline.search(window);

    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::BruteforceSearch>(bmmpy::BruteforceSearchConfig{4, 0});

    require(std::string_view(searcher->name()) == "bruteforce", "bruteforce searcher name");

    const auto dispatch_window = matrix.row_window({0, 1, 2});
    const auto actual = searcher->search(dispatch_window);
    require_same_candidates(actual, expected, "bruteforce searcher dispatch");
}

void test_cuda_bruteforce_validates_window_size() {
    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix(15, 512);

    std::vector<std::size_t> rows(15);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::CudaBruteforceSearch search;
    expect_invalid_argument([&] { (void)search.search(window); },
                            "cuda bruteforce searcher window size");
}

void test_cuda_bruteforce_validates_supported_width() {
    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix(16, 256);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::CudaBruteforceSearch search;
    expect_invalid_argument([&] { (void)search.search(window); },
                            "cuda bruteforce searcher supported width");
}

void test_cuda_bruteforce_matches_cpu_bruteforce_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix(16, 512);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{8, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    require_same_candidates(actual, expected, "cuda_bruteforce matches cpu bruteforce");
}

void test_cuda_bruteforce_supports_128_candidates_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix(16, 1024);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{128, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{128, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    require_same_candidates(actual, expected, "cuda_bruteforce supports 128 candidates");
}

void test_cuda_bruteforce_supports_4096_width_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = make_cuda_equivalence_matrix(16, 4096);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{8, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    require_same_candidates(actual, expected, "cuda_bruteforce supports 4096 width");
}

void test_sa_selector_is_deterministic_for_same_seed() {
    const bmmpy::BitMatrix matrix = make_sa_cluster_matrix();

    const bmmpy::SASelector selector({
        256,
        8,
        12345,
        bmmpy::WindowScorePolicyKind::PairwiseSynergy,
        bmmpy::CoolingPolicyKind::AdaptiveGeometric,
        32,
        0.8,
        0.99,
        1e-6,
    });

    const auto lhs = selector.select(matrix, 4);
    const auto rhs = selector.select(matrix, 4);

    require(lhs.rows == rhs.rows, "sa selector deterministic rows");
    require(lhs.score == rhs.score, "sa selector deterministic score");
    require(lhs.best_iteration == rhs.best_iteration, "sa selector deterministic best_iteration");
    require(lhs.restart_index == rhs.restart_index, "sa selector deterministic restart");
    require(lhs.accepted_moves == rhs.accepted_moves, "sa selector deterministic accepted_moves");
    require(lhs.iterations_run == rhs.iterations_run, "sa selector deterministic iterations_run");
}

void test_sa_selector_prefers_dense_cluster() {
    const bmmpy::BitMatrix matrix = make_sa_cluster_matrix();

    const bmmpy::SASelector selector({
        256,
        8,
        7,
        bmmpy::WindowScorePolicyKind::PairwiseSynergy,
        bmmpy::CoolingPolicyKind::AdaptiveGeometric,
        32,
        0.8,
        0.99,
        1e-6,
    });

    const auto result = selector.select(matrix, 4);
    require_eq<std::size_t>(result.rows, {0, 1, 2, 3}, "sa selector best cluster rows");
    require(result.score == 72, "sa selector best cluster score");

    const auto window = selector.select_window(matrix, 4);
    require_eq<std::size_t>(window.global_rows(), {0, 1, 2, 3}, "sa selector select_window rows");
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
        {"row_and_popcount_counts_intersection", &test_row_and_popcount_counts_intersection},
        {"compact_split_window_collects_expected_patterns",
         &test_compact_split_window_collects_expected_patterns},
        {"fwht_search_finds_best_candidate", &test_fwht_search_finds_best_candidate},
        {"fwht_search_respects_k", &test_fwht_search_respects_k},
        {"fwht_search_window_bounds", &test_fwht_search_window_bounds},
        {"searcher_interface_dispatch", &test_searcher_interface_dispatch},
        {"mitm_fwht_matches_fwht_search", &test_mitm_fwht_matches_fwht_search},
        {"mitm_fwht_searcher_interface_dispatch", &test_mitm_fwht_searcher_interface_dispatch},
        {"cuda_runtime_features_are_consistent", &test_cuda_runtime_features_are_consistent},
        {"cuda_mitm_fwht_searcher_interface_dispatch",
         &test_cuda_mitm_fwht_searcher_interface_dispatch},
        {"cuda_mitm_fwht_validates_window_size", &test_cuda_mitm_fwht_validates_window_size},
        {"cuda_mitm_fwht_validates_low_bits", &test_cuda_mitm_fwht_validates_low_bits},
        {"cuda_mitm_fwht_rejects_non_int16_safe_total_weight",
         &test_cuda_mitm_fwht_rejects_non_int16_safe_total_weight},
        {"cuda_mitm_fwht_matches_cpu_mitm_when_available",
         &test_cuda_mitm_fwht_matches_cpu_mitm_when_available},
        {"cuda_mitm_fwht_supports_128_candidates_when_available",
         &test_cuda_mitm_fwht_supports_128_candidates_when_available},
        {"bruteforce_searcher_interface_dispatch", &test_bruteforce_searcher_interface_dispatch},
        {"bruteforce_search_respects_k", &test_bruteforce_search_respects_k},
        {"bruteforce_search_matches_fwht_search", &test_bruteforce_search_matches_fwht_search},
        {"cuda_bruteforce_validates_window_size", &test_cuda_bruteforce_validates_window_size},
        {"cuda_bruteforce_validates_supported_width",
         &test_cuda_bruteforce_validates_supported_width},
        {"cuda_bruteforce_matches_cpu_bruteforce_when_available",
         &test_cuda_bruteforce_matches_cpu_bruteforce_when_available},
        {"cuda_bruteforce_supports_128_candidates_when_available",
         &test_cuda_bruteforce_supports_128_candidates_when_available},
        {"cuda_bruteforce_supports_4096_width_when_available",
         &test_cuda_bruteforce_supports_4096_width_when_available},
        {"sa_selector_is_deterministic_for_same_seed",
         &test_sa_selector_is_deterministic_for_same_seed},
        {"sa_selector_prefers_dense_cluster", &test_sa_selector_prefers_dense_cluster},
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
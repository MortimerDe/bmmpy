#include "bmmpy/search/bruteforce_search.hpp"
#include "bmmpy/search/cuda_bruteforce_search.hpp"
#include "bmmpy/search/cuda_mitm_fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"
#include "bmmpy/stub.hpp"
#include "test_common.hpp"

#include <numeric>
#include <vector>

using bmmpy::test::TestCase;

namespace {

void test_cuda_runtime_features_are_consistent() {
    const auto features = bmmpy::get_runtime_features();
    bmmpy::test::require(!(features.cuda_available && !features.cuda_compiled),
                         "cuda_available implies cuda_compiled");
}

void test_cuda_mitm_fwht_searcher_interface_dispatch() {
    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::CudaMitmFwhtSearch>(bmmpy::CudaMitmFwhtSearchConfig{});

    bmmpy::test::require(std::string_view(searcher->name()) == "cuda_mitm_fwht",
                         "cuda mitm searcher name");
}

void test_cuda_mitm_fwht_validates_window_size() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::CudaMitmFwhtSearch search;
    const auto window = matrix.row_window({0, 1, 2});

    bmmpy::test::expect_invalid_argument([&] { (void)search.search(window); },
                                         "cuda mitm searcher window size");
}

void test_cuda_mitm_fwht_validates_low_bits() {
    bmmpy::BitMatrix matrix(28, 4);
    matrix.set(0, 0, true);

    bmmpy::CudaMitmFwhtSearch search({64, 28});
    const auto window = matrix.row_window({
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    });

    bmmpy::test::expect_invalid_argument([&] { (void)search.search(window); },
                                         "cuda mitm searcher low_bits");
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

    bmmpy::test::expect_invalid_argument([&] { (void)search.search(window); },
                                         "cuda mitm searcher int16-safe total_weight");
}

void test_cuda_mitm_fwht_matches_cpu_mitm_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix();

    std::vector<std::size_t> rows(28);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::MitmFwhtSearch cpu(bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 8});
    bmmpy::CudaMitmFwhtSearch gpu(bmmpy::CudaMitmFwhtSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    bmmpy::test::require_same_candidates(actual, expected, "cuda_mitm_fwht matches cpu mitm");
}

void test_cuda_mitm_fwht_supports_128_candidates_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix();

    std::vector<std::size_t> rows(28);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::MitmFwhtSearch cpu(bmmpy::MitmFwhtSearchConfig{1024, 20, std::size_t{1} << 16, 128});
    bmmpy::CudaMitmFwhtSearch gpu(bmmpy::CudaMitmFwhtSearchConfig{128, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    bmmpy::test::require_same_candidates(
        actual, expected, "cuda_mitm_fwht supports 128 candidates");
}

void test_cuda_bruteforce_validates_window_size() {
    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix(15, 512);

    std::vector<std::size_t> rows(15);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::CudaBruteforceSearch search;
    bmmpy::test::expect_invalid_argument([&] { (void)search.search(window); },
                                         "cuda bruteforce searcher window size");
}

void test_cuda_bruteforce_validates_supported_width() {
    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix(16, 256);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::CudaBruteforceSearch search;
    bmmpy::test::expect_invalid_argument([&] { (void)search.search(window); },
                                         "cuda bruteforce searcher supported width");
}

void test_cuda_bruteforce_matches_cpu_bruteforce_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix(16, 512);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{8, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    bmmpy::test::require_same_candidates(
        actual, expected, "cuda_bruteforce matches cpu bruteforce");
}

void test_cuda_bruteforce_supports_128_candidates_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix(16, 1024);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{128, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{128, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    bmmpy::test::require_same_candidates(
        actual, expected, "cuda_bruteforce supports 128 candidates");
}

void test_cuda_bruteforce_supports_4096_width_when_available() {
    const auto features = bmmpy::get_runtime_features();
    if (!(features.cuda_compiled && features.cuda_available))
        return;

    const bmmpy::BitMatrix matrix = bmmpy::test::make_cuda_equivalence_matrix(16, 4096);

    std::vector<std::size_t> rows(16);
    std::iota(rows.begin(), rows.end(), 0);
    const auto window = matrix.row_window(rows);

    bmmpy::BruteforceSearch cpu(bmmpy::BruteforceSearchConfig{8, 0});
    bmmpy::CudaBruteforceSearch gpu(bmmpy::CudaBruteforceSearchConfig{8, 0});

    const auto expected = cpu.search(window);
    const auto actual = gpu.search(window);

    bmmpy::test::require_same_candidates(actual, expected, "cuda_bruteforce supports 4096 width");
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
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
        {"cuda_bruteforce_validates_window_size", &test_cuda_bruteforce_validates_window_size},
        {"cuda_bruteforce_validates_supported_width",
         &test_cuda_bruteforce_validates_supported_width},
        {"cuda_bruteforce_matches_cpu_bruteforce_when_available",
         &test_cuda_bruteforce_matches_cpu_bruteforce_when_available},
        {"cuda_bruteforce_supports_128_candidates_when_available",
         &test_cuda_bruteforce_supports_128_candidates_when_available},
        {"cuda_bruteforce_supports_4096_width_when_available",
         &test_cuda_bruteforce_supports_4096_width_when_available},
    };

    return bmmpy::test::run_tests(tests);
}
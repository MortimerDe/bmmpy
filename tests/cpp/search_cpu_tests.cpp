#include "bmmpy/search/bruteforce_search.hpp"
#include "bmmpy/search/fwht_search.hpp"
#include "bmmpy/search/mitm_fwht_search.hpp"
#include "bmmpy/search/sa_selector.hpp"
#include "bmmpy/search/searcher.hpp"
#include "bmmpy/search/split_window_prep.hpp"
#include "test_common.hpp"

#include <memory>
#include <vector>

using bmmpy::test::TestCase;

void append_sa_selector_tests(std::vector<TestCase>& tests);

namespace {

void test_fwht_search_finds_best_candidate() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    bmmpy::FwhtSearch search;
    const auto window = matrix.row_window({0, 1});
    const auto candidates = search.search(window);

    bmmpy::test::require(candidates.size() == 3, "fwht_search candidate count");
    bmmpy::test::require(candidates[0].mask_u64() == 0x3ull, "fwht_search best mask");
    bmmpy::test::require(candidates[0].weight == 0, "fwht_search best weight");

    for (std::size_t i = 1; i < candidates.size(); ++i)
        bmmpy::test::require(candidates[i - 1].weight <= candidates[i].weight,
                             "fwht_search result order");
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

    bmmpy::test::require(candidates.size() == 1, "fwht_search k limit");
    bmmpy::test::require(candidates[0].mask_u64() == 0x3ull, "fwht_search k best mask");
    bmmpy::test::require(candidates[0].weight == 0, "fwht_search k best weight");
}

void test_fwht_search_window_bounds() {
    bmmpy::BitMatrix matrix(1, 1);
    bmmpy::FwhtSearch search;

    bmmpy::test::expect_out_of_range([&] { (void)matrix.row_window({1}); }, "row_window bounds");
}

void test_searcher_interface_dispatch() {
    bmmpy::BitMatrix matrix(2, 5);
    for (std::size_t col : {0u, 2u, 4u}) {
        matrix.set(0, col, true);
        matrix.set(1, col, true);
    }

    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::FwhtSearch>(bmmpy::FwhtSearchConfig{16, 1});

    bmmpy::test::require(std::string_view(searcher->name()) == "fwht", "searcher name");

    const auto window = matrix.row_window({0, 1});
    const auto candidates = searcher->search(window);
    bmmpy::test::require(candidates.size() == 1, "searcher candidate count");
    bmmpy::test::require(candidates[0].mask_u64() == 0x3ull, "searcher best mask");
    bmmpy::test::require(candidates[0].weight == 0, "searcher best weight");
}

void test_mitm_fwht_matches_fwht_search() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
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

    bmmpy::test::require_same_candidates(actual, expected, "mitm_fwht matches fwht");
}

void test_mitm_fwht_searcher_interface_dispatch() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::FwhtSearch baseline({16, 4});
    const auto window = matrix.row_window({0, 1, 2});
    const auto expected = baseline.search(window);

    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::MitmFwhtSearch>(bmmpy::MitmFwhtSearchConfig{
            1024,
            20,
            std::size_t{1} << 16,
            4,
        });

    bmmpy::test::require(std::string_view(searcher->name()) == "mitm_fwht", "mitm searcher name");

    const auto dispatch_window = matrix.row_window({0, 1, 2});
    const auto actual = searcher->search(dispatch_window);
    bmmpy::test::require_same_candidates(actual, expected, "mitm searcher dispatch");
}

void test_compact_split_window_collects_expected_patterns() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
        "11001",
        "00100",
        "01101",
        "01001",
    });

    const auto window = matrix.row_window({0, 1, 2, 3});
    const auto compact = bmmpy::build_compact_split_window(window, 2);

    bmmpy::test::require(compact.t == 4, "compact_split_window t");
    bmmpy::test::require(compact.low_bits == 2, "compact_split_window low_bits");
    bmmpy::test::require(compact.high_bits == 2, "compact_split_window high_bits");
    bmmpy::test::require(compact.total_weight == 4, "compact_split_window total_weight");

    bmmpy::test::require_eq<std::uint64_t>(compact.q,
                                           {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}},
                                           "compact_split_window q");
    bmmpy::test::require_eq<std::uint64_t>(compact.r,
                                           {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{1}},
                                           "compact_split_window r");
    bmmpy::test::require_eq<std::int32_t>(
        compact.multiplicity, {1, 1, 2}, "compact_split_window multiplicity");
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

    bmmpy::test::require(candidates.size() == 1, "bruteforce_search k limit");
    bmmpy::test::require(candidates[0].mask_u64() == 0x3ull, "bruteforce_search k best mask");
    bmmpy::test::require(candidates[0].weight == 0, "bruteforce_search k best weight");
}

void test_bruteforce_search_matches_fwht_search() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
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

    bmmpy::test::require_same_candidates(actual, expected, "bruteforce matches fwht");
}

void test_bruteforce_searcher_interface_dispatch() {
    const bmmpy::BitMatrix matrix = bmmpy::test::matrix_from_rows({
        "10101",
        "11100",
        "00111",
    });

    bmmpy::FwhtSearch baseline({16, 4});
    const auto window = matrix.row_window({0, 1, 2});
    const auto expected = baseline.search(window);

    std::unique_ptr<bmmpy::Searcher> searcher =
        std::make_unique<bmmpy::BruteforceSearch>(bmmpy::BruteforceSearchConfig{4, 0});

    bmmpy::test::require(std::string_view(searcher->name()) == "bruteforce",
                         "bruteforce searcher name");

    const auto dispatch_window = matrix.row_window({0, 1, 2});
    const auto actual = searcher->search(dispatch_window);
    bmmpy::test::require_same_candidates(actual, expected, "bruteforce searcher dispatch");
}

} // namespace

void append_search_cpu_tests(std::vector<TestCase>& tests) {
    tests.push_back({"compact_split_window_collects_expected_patterns",
                     &test_compact_split_window_collects_expected_patterns});
    tests.push_back({"fwht_search_finds_best_candidate", &test_fwht_search_finds_best_candidate});
    tests.push_back({"fwht_search_respects_k", &test_fwht_search_respects_k});
    tests.push_back({"fwht_search_window_bounds", &test_fwht_search_window_bounds});
    tests.push_back({"searcher_interface_dispatch", &test_searcher_interface_dispatch});
    tests.push_back({"mitm_fwht_matches_fwht_search", &test_mitm_fwht_matches_fwht_search});
    tests.push_back(
        {"mitm_fwht_searcher_interface_dispatch", &test_mitm_fwht_searcher_interface_dispatch});
    tests.push_back({"bruteforce_search_respects_k", &test_bruteforce_search_respects_k});
    tests.push_back(
        {"bruteforce_search_matches_fwht_search", &test_bruteforce_search_matches_fwht_search});
    tests.push_back(
        {"bruteforce_searcher_interface_dispatch", &test_bruteforce_searcher_interface_dispatch});
}

int main() {
    std::vector<TestCase> tests;
    append_search_cpu_tests(tests);
    append_sa_selector_tests(tests);
    return bmmpy::test::run_tests(tests);
}
#include "bmmpy/search/sa_selector.hpp"
#include "test_common.hpp"

#include <vector>

using bmmpy::test::TestCase;

namespace {

void test_sa_selector_is_deterministic_for_same_seed() {
    const bmmpy::BitMatrix matrix = bmmpy::test::make_sa_cluster_matrix();

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

    bmmpy::test::require(lhs.rows == rhs.rows, "sa selector deterministic rows");
    bmmpy::test::require(lhs.score == rhs.score, "sa selector deterministic score");
    bmmpy::test::require(lhs.best_iteration == rhs.best_iteration,
                         "sa selector deterministic best_iteration");
    bmmpy::test::require(lhs.restart_index == rhs.restart_index,
                         "sa selector deterministic restart");
    bmmpy::test::require(lhs.accepted_moves == rhs.accepted_moves,
                         "sa selector deterministic accepted_moves");
    bmmpy::test::require(lhs.iterations_run == rhs.iterations_run,
                         "sa selector deterministic iterations_run");
}

void test_sa_selector_prefers_dense_cluster() {
    const bmmpy::BitMatrix matrix = bmmpy::test::make_sa_cluster_matrix();

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
    bmmpy::test::require_eq<std::size_t>(
        result.rows, {0, 1, 2, 3}, "sa selector best cluster rows");
    bmmpy::test::require(result.score == 72, "sa selector best cluster score");

    const auto window = selector.select_window(matrix, 4);
    bmmpy::test::require_eq<std::size_t>(
        window.global_rows(), {0, 1, 2, 3}, "sa selector select_window rows");
}

} // namespace

void append_sa_selector_tests(std::vector<TestCase>& tests) {
    tests.push_back({"sa_selector_is_deterministic_for_same_seed",
                     &test_sa_selector_is_deterministic_for_same_seed});
    tests.push_back({"sa_selector_prefers_dense_cluster", &test_sa_selector_prefers_dense_cluster});
}
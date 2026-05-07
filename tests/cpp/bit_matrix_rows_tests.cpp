#include "test_common.hpp"

#include <vector>

using bmmpy::BitMatrix;
using bmmpy::test::TestCase;

namespace {

void test_row_operations() {
    BitMatrix matrix = bmmpy::test::matrix_from_rows({"1011", "0110"});
    matrix.row_xor(0, 1);
    bmmpy::test::expect_rows(matrix, {"1101", "0110"});

    BitMatrix source = bmmpy::test::matrix_from_rows({"0000", "1111"});
    matrix.row_xor_from(1, source, 1);
    bmmpy::test::expect_rows(matrix, {"1101", "1001"});

    matrix.swap_rows(0, 1);
    bmmpy::test::expect_rows(matrix, {"1001", "1101"});
}

void test_rank_and_row_selection() {
    BitMatrix dependent = bmmpy::test::matrix_from_rows({"1100", "0110", "1010"});
    bmmpy::test::require(dependent.rank() == 2, "rank of dependent matrix should be 2");

    BitMatrix full_rank = bmmpy::test::matrix_from_rows({"100", "010", "001"});
    bmmpy::test::require(full_rank.rank() == 3, "rank of identity matrix should be 3");

    BitMatrix source = bmmpy::test::matrix_from_rows({
        "10000",
        "01000",
        "00100",
        "00010",
    });

    BitMatrix extracted = source.extract_rows_by_indices({3, 1});
    bmmpy::test::expect_rows(extracted, {"00010", "01000"});

    BitMatrix target = bmmpy::test::matrix_from_rows({
        "00000",
        "11111",
        "00000",
        "11111",
    });

    target.insert_rows_by_indices(extracted, {0, 2});
    bmmpy::test::expect_rows(target, {"00010", "11111", "01000", "11111"});
}

void test_row_window_view() {
    BitMatrix source = bmmpy::test::matrix_from_rows({
        "10000",
        "01000",
        "00100",
        "00010",
    });

    auto window = source.row_window({3, 1});
    bmmpy::test::require(window.size() == 2, "row window size mismatch");
    bmmpy::test::require(window.cols() == 5, "row window cols mismatch");
    bmmpy::test::require(window.global_row(0) == 3, "row window global row 0 mismatch");
    bmmpy::test::require(window.global_row(1) == 1, "row window global row 1 mismatch");
    bmmpy::test::require(window.row_popcount(0) == 1, "row window row_popcount mismatch");
    bmmpy::test::expect_rows(window.materialize(), {"00010", "01000"});

    window.row_xor(0, 1);
    bmmpy::test::expect_rows(source, {"10000", "01000", "00100", "01010"});

    bmmpy::test::expect_out_of_range([&] { (void)source.row_window({4}); },
                                     "row window out of bounds");
    bmmpy::test::expect_invalid_argument([&] { (void)source.row_window({1, 1}); },
                                         "row window duplicate rows");
}

void test_add_mul_and_power() {
    BitMatrix lhs = bmmpy::test::matrix_from_rows({"1010", "0110"});
    BitMatrix rhs = bmmpy::test::matrix_from_rows({"1100", "0101"});
    bmmpy::test::expect_rows(lhs.add(rhs), {"0110", "0011"});

    BitMatrix mul_lhs = bmmpy::test::matrix_from_rows({"101", "011"});
    BitMatrix mul_rhs = bmmpy::test::matrix_from_rows({"10", "11", "01"});
    bmmpy::test::expect_rows(mul_lhs.mul(mul_rhs), {"11", "10"});

    BitMatrix fib = bmmpy::test::matrix_from_rows({"11", "10"});
    bmmpy::test::expect_rows(BitMatrix::identity(2), {"10", "01"});
    bmmpy::test::expect_rows(fib.power(0), {"10", "01"});
    bmmpy::test::expect_rows(fib.power(2), {"01", "11"});
    bmmpy::test::expect_rows(fib.power(3), {"10", "01"});
}

void test_row_window_total_weight() {
    BitMatrix source = bmmpy::test::matrix_from_rows({
        "10101",
        "00101",
        "00010",
    });

    const auto window = source.row_window({0, 1, 2});

    bmmpy::test::require(window.row_popcount(0) == 3, "row window row_popcount(0) mismatch");
    bmmpy::test::require(window.row_popcount(1) == 2, "row window row_popcount(1) mismatch");
    bmmpy::test::require(window.row_popcount(2) == 1, "row window row_popcount(2) mismatch");
    bmmpy::test::require(window.total_weight() == 4, "row window total_weight mismatch");
}

} // namespace

void append_bit_matrix_rows_tests(std::vector<TestCase>& tests) {
    tests.push_back({"row_operations", &test_row_operations});
    tests.push_back({"add_mul_and_power", &test_add_mul_and_power});
    tests.push_back({"rank_and_row_selection", &test_rank_and_row_selection});
    tests.push_back({"row_window_view", &test_row_window_view});
    tests.push_back({"row_window_total_weight", &test_row_window_total_weight});
}
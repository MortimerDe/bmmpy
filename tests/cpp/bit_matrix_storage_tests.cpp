#include "test_common.hpp"

#include <array>
#include <cstdint>
#include <vector>

using bmmpy::BitMatrix;
using bmmpy::MatrixErr;
using bmmpy::test::TestCase;

void append_bit_matrix_rows_tests(std::vector<TestCase>& tests);
void append_bit_matrix_io_tests(std::vector<TestCase>& tests);
void append_bit_matrix_hash_tests(std::vector<TestCase>& tests);

namespace {

void test_shape_and_storage() {
    BitMatrix matrix(3, 130);

    bmmpy::test::require(matrix.rows() == 3, "rows mismatch");
    bmmpy::test::require(matrix.cols() == 130, "cols mismatch");
    bmmpy::test::require(matrix.words_per_row() == 3, "words_per_row mismatch");
    bmmpy::test::require(matrix.stride_words() == 4, "stride should be padded to 4 words");
    bmmpy::test::require(matrix.total_words() == 12, "total_words mismatch");
    bmmpy::test::require(matrix.total_bytes() == 12 * sizeof(std::uint64_t),
                         "total_bytes mismatch");
    bmmpy::test::require(reinterpret_cast<std::uintptr_t>(matrix.data()) % BitMatrix::k_alignment ==
                             0,
                         "data pointer is not aligned");
    bmmpy::test::require(matrix.weight() == 0, "new matrix must be zero-initialized");

    matrix.set(0, 0, true);
    matrix.set(1, 63, true);
    matrix.set(1, 64, true);
    matrix.set(2, 129, true);

    bmmpy::test::require(matrix.get(0, 0), "bit (0,0) should be set");
    bmmpy::test::require(matrix.get(1, 63), "bit (1,63) should be set");
    bmmpy::test::require(matrix.get(1, 64), "bit (1,64) should be set");
    bmmpy::test::require(matrix.get(2, 129), "bit (2,129) should be set");
    bmmpy::test::require(!matrix.get(0, 1), "bit (0,1) should be clear");
    bmmpy::test::require(matrix.row_popcount(1) == 2, "row_popcount mismatch");
    bmmpy::test::require(matrix.weight() == 4, "weight mismatch");
    bmmpy::test::require(matrix.row_words(0)[3] == 0, "padding word should remain zero");
}

void test_copy_and_move_semantics() {
    BitMatrix original = bmmpy::test::matrix_from_rows({"1010", "0101"});

    BitMatrix copied(original);
    original.set(0, 0, false);
    bmmpy::test::expect_rows(copied, {"1010", "0101"});

    BitMatrix assigned;
    assigned = copied;
    copied.set(1, 0, true);
    bmmpy::test::expect_rows(assigned, {"1010", "0101"});

    BitMatrix moved(std::move(assigned));
    bmmpy::test::expect_rows(moved, {"1010", "0101"});
    bmmpy::test::require(assigned.data() == nullptr, "moved-from matrix should release data");
    bmmpy::test::require(assigned.rows() == 0, "moved-from matrix should reset rows");
    bmmpy::test::require(assigned.cols() == 0, "moved-from matrix should reset cols");

    BitMatrix move_assigned;
    move_assigned = std::move(moved);
    bmmpy::test::expect_rows(move_assigned, {"1010", "0101"});
    bmmpy::test::require(moved.data() == nullptr, "move-assigned source should release data");
    bmmpy::test::require(moved.rows() == 0, "move-assigned source should reset rows");
    bmmpy::test::require(moved.cols() == 0, "move-assigned source should reset cols");
}

void test_copy_from_words_and_exceptions() {
    BitMatrix matrix(2, 2);
    std::array<std::uint64_t, 8> words{};
    words[0] = 0b01;
    words[4] = 0b10;

    matrix.copy_from_words(words.data(), words.size());
    bmmpy::test::expect_rows(matrix, {"10", "01"});

    bmmpy::test::expect_out_of_range([&] { matrix.set(2, 0, true); }, "set out of bounds");
    bmmpy::test::expect_out_of_range([&] { (void)matrix.get(0, 2); }, "get out of bounds");
    bmmpy::test::expect_out_of_range([&] { (void)matrix.row_words(2); }, "row_words out of bounds");
    bmmpy::test::expect_out_of_range([&] { matrix.extract_rows_by_indices({2}); },
                                     "extract_rows_by_indices out of bounds");

    std::array<std::uint64_t, 1> wrong_words{};
    bmmpy::test::expect_matrix_error(
        [&] { matrix.copy_from_words(wrong_words.data(), wrong_words.size()); },
        MatrixErr::DimensionMismatch,
        "copy_from_words");

    BitMatrix add_mismatch(3, 2);
    bmmpy::test::expect_matrix_error([&] { (void)matrix.add(add_mismatch); },
                                     MatrixErr::DimensionMismatch,
                                     "add dimension mismatch");

    BitMatrix mul_mismatch(4, 1);
    bmmpy::test::expect_matrix_error([&] { (void)matrix.mul(mul_mismatch); },
                                     MatrixErr::DimensionMismatch,
                                     "mul dimension mismatch");

    BitMatrix nonsquare(2, 3);
    bmmpy::test::expect_matrix_error([&] { (void)nonsquare.power(2); },
                                     MatrixErr::DimensionMismatch,
                                     "power on nonsquare matrix");

    BitMatrix different_cols(2, 3);
    bmmpy::test::expect_matrix_error([&] { matrix.row_xor_from(0, different_cols, 0); },
                                     MatrixErr::DimensionMismatch,
                                     "row_xor_from dimension mismatch");

    BitMatrix one_row(1, 2);
    bmmpy::test::expect_matrix_error([&] { matrix.insert_rows_by_indices(one_row, {0, 1}); },
                                     MatrixErr::DimensionMismatch,
                                     "insert_rows_by_indices row count mismatch");

    bmmpy::test::expect_out_of_range([&] { matrix.insert_rows_by_indices(one_row, {2}); },
                                     "insert_rows_by_indices target out of bounds");
}

} // namespace

void append_bit_matrix_storage_tests(std::vector<TestCase>& tests) {
    tests.push_back({"shape_and_storage", &test_shape_and_storage});
    tests.push_back({"copy_and_move_semantics", &test_copy_and_move_semantics});
    tests.push_back({"copy_from_words_and_exceptions", &test_copy_from_words_and_exceptions});
}

int main() {
    std::vector<TestCase> tests;
    append_bit_matrix_storage_tests(tests);
    append_bit_matrix_rows_tests(tests);
    append_bit_matrix_io_tests(tests);
    append_bit_matrix_hash_tests(tests);
    return bmmpy::test::run_tests(tests);
}
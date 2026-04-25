#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using bmmpy::BitMatrix;
using bmmpy::MatrixErr;
using bmmpy::MatrixError;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool condition, std::string_view message) {
    if (!condition)
        fail(std::string(message));
}

BitMatrix matrix_from_rows(std::initializer_list<std::string_view> rows) {
    const std::size_t row_count = rows.size();
    const std::size_t col_count = row_count == 0 ? 0 : rows.begin()->size();

    BitMatrix matrix(row_count, col_count);

    std::size_t row_index = 0;
    for (std::string_view row : rows) {
        require(row.size() == col_count, "row width mismatch in test fixture");

        for (std::size_t col_index = 0; col_index < col_count; ++col_index) {
            const char ch = row[col_index];
            if (ch == '1') {
                matrix.set(row_index, col_index, true);
            } else if (ch != '0') {
                fail("test fixture must contain only '0' or '1'");
            }
        }

        ++row_index;
    }

    return matrix;
}

void expect_rows(const BitMatrix& matrix, std::initializer_list<std::string_view> expected_rows) {
    const std::size_t expected_row_count = expected_rows.size();
    const std::size_t expected_col_count =
        expected_row_count == 0 ? 0 : expected_rows.begin()->size();

    require(matrix.rows() == expected_row_count, "unexpected row count");
    require(matrix.cols() == expected_col_count, "unexpected column count");

    std::size_t row_index = 0;
    for (std::string_view row : expected_rows) {
        require(row.size() == expected_col_count, "row width mismatch in expected matrix");

        for (std::size_t col_index = 0; col_index < expected_col_count; ++col_index) {
            const bool expected = row[col_index] == '1';
            const bool actual = matrix.get(row_index, col_index);
            if (actual != expected) {
                fail("matrix contents mismatch");
            }
        }

        ++row_index;
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

template <typename Fn>
void expect_matrix_error(Fn&& fn, MatrixErr expected_code, std::string_view context) {
    try {
        fn();
    } catch (const MatrixError& ex) {
        require(ex.code() == expected_code, std::string(context) + ": MatrixError code mismatch");
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected MatrixError, got: " + ex.what());
    }

    fail(std::string(context) + ": expected MatrixError");
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

std::filesystem::path unique_test_path(std::string_view suffix) {
    static std::uint64_t counter = 0;
    return std::filesystem::temp_directory_path() /
           ("bmmpy_bit_matrix_" + std::to_string(counter++) + "_" + std::string(suffix));
}

void test_shape_and_storage() {
    BitMatrix matrix(3, 130);

    require(matrix.rows() == 3, "rows mismatch");
    require(matrix.cols() == 130, "cols mismatch");
    require(matrix.words_per_row() == 3, "words_per_row mismatch");
    require(matrix.stride_words() == 4, "stride should be padded to 4 words");
    require(matrix.total_words() == 12, "total_words mismatch");
    require(matrix.total_bytes() == 12 * sizeof(std::uint64_t), "total_bytes mismatch");
    require(reinterpret_cast<std::uintptr_t>(matrix.data()) % BitMatrix::k_alignment == 0,
            "data pointer is not aligned");
    require(matrix.weight() == 0, "new matrix must be zero-initialized");

    matrix.set(0, 0, true);
    matrix.set(1, 63, true);
    matrix.set(1, 64, true);
    matrix.set(2, 129, true);

    require(matrix.get(0, 0), "bit (0,0) should be set");
    require(matrix.get(1, 63), "bit (1,63) should be set");
    require(matrix.get(1, 64), "bit (1,64) should be set");
    require(matrix.get(2, 129), "bit (2,129) should be set");
    require(!matrix.get(0, 1), "bit (0,1) should be clear");
    require(matrix.row_popcount(1) == 2, "row_popcount mismatch");
    require(matrix.weight() == 4, "weight mismatch");
    require(matrix.row_words(0)[3] == 0, "padding word should remain zero");
}

void test_copy_and_move_semantics() {
    BitMatrix original = matrix_from_rows({"1010", "0101"});

    BitMatrix copied(original);
    original.set(0, 0, false);
    expect_rows(copied, {"1010", "0101"});

    BitMatrix assigned;
    assigned = copied;
    copied.set(1, 0, true);
    expect_rows(assigned, {"1010", "0101"});

    BitMatrix moved(std::move(assigned));
    expect_rows(moved, {"1010", "0101"});
    require(assigned.data() == nullptr, "moved-from matrix should release data");
    require(assigned.rows() == 0, "moved-from matrix should reset rows");
    require(assigned.cols() == 0, "moved-from matrix should reset cols");

    BitMatrix move_assigned;
    move_assigned = std::move(moved);
    expect_rows(move_assigned, {"1010", "0101"});
    require(moved.data() == nullptr, "move-assigned source should release data");
    require(moved.rows() == 0, "move-assigned source should reset rows");
    require(moved.cols() == 0, "move-assigned source should reset cols");
}

void test_row_operations() {
    BitMatrix matrix = matrix_from_rows({"1011", "0110"});
    matrix.row_xor(0, 1);
    expect_rows(matrix, {"1101", "0110"});

    BitMatrix source = matrix_from_rows({"0000", "1111"});
    matrix.row_xor_from(1, source, 1);
    expect_rows(matrix, {"1101", "1001"});

    matrix.swap_rows(0, 1);
    expect_rows(matrix, {"1001", "1101"});
}

void test_add_mul_and_power() {
    BitMatrix lhs = matrix_from_rows({"1010", "0110"});
    BitMatrix rhs = matrix_from_rows({"1100", "0101"});
    expect_rows(lhs.add(rhs), {"0110", "0011"});

    BitMatrix mul_lhs = matrix_from_rows({"101", "011"});
    BitMatrix mul_rhs = matrix_from_rows({"10", "11", "01"});
    expect_rows(mul_lhs.mul(mul_rhs), {"11", "10"});

    BitMatrix fib = matrix_from_rows({"11", "10"});
    expect_rows(BitMatrix::identity(2), {"10", "01"});
    expect_rows(fib.power(0), {"10", "01"});
    expect_rows(fib.power(2), {"01", "11"});
    expect_rows(fib.power(3), {"10", "01"});
}

void test_rank_and_row_selection() {
    BitMatrix dependent = matrix_from_rows({"1100", "0110", "1010"});
    require(dependent.rank() == 2, "rank of dependent matrix should be 2");

    BitMatrix full_rank = matrix_from_rows({"100", "010", "001"});
    require(full_rank.rank() == 3, "rank of identity matrix should be 3");

    BitMatrix source = matrix_from_rows({
        "10000",
        "01000",
        "00100",
        "00010",
    });

    BitMatrix extracted = source.extract_rows_by_indices({3, 1});
    expect_rows(extracted, {"00010", "01000"});

    BitMatrix target = matrix_from_rows({
        "00000",
        "11111",
        "00000",
        "11111",
    });

    target.insert_rows_by_indices(extracted, {0, 2});
    expect_rows(target, {"00010", "11111", "01000", "11111"});
}

void test_row_window_view() {
    BitMatrix source = matrix_from_rows({
        "10000",
        "01000",
        "00100",
        "00010",
    });

    auto window = source.row_window({3, 1});
    require(window.size() == 2, "row window size mismatch");
    require(window.cols() == 5, "row window cols mismatch");
    require(window.global_row(0) == 3, "row window global row 0 mismatch");
    require(window.global_row(1) == 1, "row window global row 1 mismatch");
    require(window.row_popcount(0) == 1, "row window row_popcount mismatch");
    expect_rows(window.materialize(), {"00010", "01000"});

    window.row_xor(0, 1);
    expect_rows(source, {"10000", "01000", "00100", "01010"});

    expect_out_of_range([&] { (void)source.row_window({4}); }, "row window out of bounds");
    expect_invalid_argument([&] { (void)source.row_window({1, 1}); }, "row window duplicate rows");
}

void test_copy_from_words_and_exceptions() {
    BitMatrix matrix(2, 2);
    std::array<std::uint64_t, 8> words{};
    words[0] = 0b01;
    words[4] = 0b10;

    matrix.copy_from_words(words.data(), words.size());
    expect_rows(matrix, {"10", "01"});

    expect_out_of_range([&] { matrix.set(2, 0, true); }, "set out of bounds");
    expect_out_of_range([&] { (void)matrix.get(0, 2); }, "get out of bounds");
    expect_out_of_range([&] { (void)matrix.row_words(2); }, "row_words out of bounds");
    expect_out_of_range([&] { matrix.extract_rows_by_indices({2}); },
                        "extract_rows_by_indices out of bounds");

    std::array<std::uint64_t, 1> wrong_words{};
    expect_matrix_error([&] { matrix.copy_from_words(wrong_words.data(), wrong_words.size()); },
                        MatrixErr::DimensionMismatch,
                        "copy_from_words");

    BitMatrix add_mismatch(3, 2);
    expect_matrix_error([&] { (void)matrix.add(add_mismatch); },
                        MatrixErr::DimensionMismatch,
                        "add dimension mismatch");

    BitMatrix mul_mismatch(4, 1);
    expect_matrix_error([&] { (void)matrix.mul(mul_mismatch); },
                        MatrixErr::DimensionMismatch,
                        "mul dimension mismatch");

    BitMatrix nonsquare(2, 3);
    expect_matrix_error([&] { (void)nonsquare.power(2); },
                        MatrixErr::DimensionMismatch,
                        "power on nonsquare matrix");

    BitMatrix different_cols(2, 3);
    expect_matrix_error([&] { matrix.row_xor_from(0, different_cols, 0); },
                        MatrixErr::DimensionMismatch,
                        "row_xor_from dimension mismatch");

    BitMatrix one_row(1, 2);
    expect_matrix_error([&] { matrix.insert_rows_by_indices(one_row, {0, 1}); },
                        MatrixErr::DimensionMismatch,
                        "insert_rows_by_indices row count mismatch");

    expect_out_of_range([&] { matrix.insert_rows_by_indices(one_row, {2}); },
                        "insert_rows_by_indices target out of bounds");
}

void test_text_io_roundtrip() {
    BitMatrix original = matrix_from_rows({
        "10101",
        "01010",
        "11100",
    });

    std::stringstream buffer;
    original.save_text(buffer);

    BitMatrix loaded = BitMatrix::load_text(buffer);
    expect_rows(loaded, {"10101", "01010", "11100"});
}

void test_binary_io_roundtrip_with_unpadded_payload() {
    BitMatrix original(3, 130);
    original.set(0, 0, true);
    original.set(0, 64, true);
    original.set(1, 63, true);
    original.set(1, 129, true);
    original.set(2, 5, true);

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    original.save_binary(buffer);
    buffer.seekg(0);

    BitMatrix loaded = BitMatrix::load_binary(buffer);

    require(loaded.rows() == 3, "binary io rows mismatch");
    require(loaded.cols() == 130, "binary io cols mismatch");
    require(loaded.words_per_row() == 3, "binary io words_per_row mismatch");
    expect_rows(loaded,
                {
                    "100000000000000000000000000000000000000000000000000000000000000010000000000000"
                    "0000000000000000000000000000000000000000000000000000",
                    "000000000000000000000000000000000000000000000000000000000000000100000000000000"
                    "0000000000000000000000000000000000000000000000000001",
                    "000001000000000000000000000000000000000000000000000000000000000000000000000000"
                    "0000000000000000000000000000000000000000000000000000",
                });
}

void test_text_io_exceptions() {
    {
        std::stringstream buffer("2 3\n010\n01x\n");
        try {
            (void)BitMatrix::load_text(buffer);
            fail("expected text io failure");
        } catch (const std::runtime_error&) {
        }
    }

    {
        std::stringstream buffer("2 3\n010\n01\n");
        try {
            (void)BitMatrix::load_text(buffer);
            fail("expected text io failure");
        } catch (const std::runtime_error&) {
        }
    }
}

void test_binary_io_exceptions() {
    {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        buffer.write("BADMAGC", 7);
        buffer.seekg(0);

        try {
            (void)BitMatrix::load_binary(buffer);
            fail("expected binary io failure");
        } catch (const std::runtime_error&) {
        }
    }

    {
        std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
        BitMatrix matrix = matrix_from_rows({"101", "010"});
        matrix.save_binary(buffer);
        std::string data = buffer.str();
        data.resize(data.size() - 1);

        std::stringstream truncated(data, std::ios::in | std::ios::out | std::ios::binary);
        try {
            (void)BitMatrix::load_binary(truncated);
            fail("expected truncated binary io failure");
        } catch (const std::runtime_error&) {
        }
    }
}

void test_row_window_total_weight() {
    BitMatrix source = matrix_from_rows({
        "10101",
        "00101",
        "00010",
    });

    const auto window = source.row_window({0, 1, 2});

    require(window.row_popcount(0) == 3, "row window row_popcount(0) mismatch");
    require(window.row_popcount(1) == 2, "row window row_popcount(1) mismatch");
    require(window.row_popcount(2) == 1, "row window row_popcount(2) mismatch");
    require(window.total_weight() == 4, "row window total_weight mismatch");
}

struct TestCase {
    const char* name;
    void (*fn)();
};

} // namespace

int main() {
    const TestCase tests[] = {
        {"shape_and_storage", &test_shape_and_storage},
        {"copy_and_move_semantics", &test_copy_and_move_semantics},
        {"row_operations", &test_row_operations},
        {"add_mul_and_power", &test_add_mul_and_power},
        {"rank_and_row_selection", &test_rank_and_row_selection},
        {"copy_from_words_and_exceptions", &test_copy_from_words_and_exceptions},
        {"text_io_roundtrip", &test_text_io_roundtrip},
        {"binary_io_roundtrip_with_unpadded_payload",
         &test_binary_io_roundtrip_with_unpadded_payload},
        {"text_io_exceptions", &test_text_io_exceptions},
        {"binary_io_exceptions", &test_binary_io_exceptions},
        {"row_window_view", &test_row_window_view},
        {"row_window_view", &test_row_window_view},
        {"row_window_total_weight", &test_row_window_total_weight},
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

    std::cout << "All BitMatrix tests passed\n";
    return 0;
}
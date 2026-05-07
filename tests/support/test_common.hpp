#pragma once

#include "bmmpy/core/bit_matrix.hpp"
#include "bmmpy/core/row_window.hpp"
#include "bmmpy/types/candidate.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace bmmpy::test {

struct TestCase {
    const char* name;
    void (*fn)();
};

[[noreturn]] inline void fail(const std::string& message) { throw std::runtime_error(message); }

inline void require(bool condition, std::string_view message) {
    if (!condition)
        fail(std::string(message));
}

inline BitMatrix matrix_from_rows(std::initializer_list<std::string_view> rows) {
    const std::size_t row_count = rows.size();
    const std::size_t col_count = row_count == 0 ? 0 : rows.begin()->size();

    BitMatrix matrix(row_count, col_count);

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

inline void expect_rows(const BitMatrix& matrix,
                        std::initializer_list<std::string_view> expected_rows) {
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
            if (actual != expected)
                fail("matrix contents mismatch");
        }

        ++row_index;
    }
}

template <typename T>
inline void require_eq(const std::vector<T>& actual,
                       std::initializer_list<T> expected,
                       std::string_view context) {
    require(actual.size() == expected.size(), std::string(context) + ": size mismatch");

    auto it = expected.begin();
    for (std::size_t i = 0; i < actual.size(); ++i, ++it) {
        if (actual[i] != *it)
            fail(std::string(context) + ": value mismatch");
    }
}

inline void require_same_candidates(const std::vector<bmmpy::Candidate>& actual,
                                    const std::vector<bmmpy::Candidate>& expected,
                                    std::string_view context) {
    require(actual.size() == expected.size(), std::string(context) + ": size mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(actual[i].mask_u64() == expected[i].mask_u64(),
                std::string(context) + ": mask mismatch");
        require(actual[i].weight == expected[i].weight, std::string(context) + ": weight mismatch");
    }
}

template <typename Fn> inline void expect_out_of_range(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::out_of_range&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::out_of_range, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::out_of_range");
}

template <typename Fn> inline void expect_invalid_argument(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::invalid_argument, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::invalid_argument");
}

template <typename Fn> inline void expect_runtime_error(Fn&& fn, std::string_view context) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return;
    } catch (const std::exception& ex) {
        fail(std::string(context) + ": expected std::runtime_error, got: " + ex.what());
    }

    fail(std::string(context) + ": expected std::runtime_error");
}

template <typename Fn>
inline void expect_matrix_error(Fn&& fn, MatrixErr expected_code, std::string_view context) {
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

inline std::filesystem::path unique_test_path(std::string_view suffix) {
    static std::uint64_t counter = 0;
    return std::filesystem::temp_directory_path() /
           ("bmmpy_test_" + std::to_string(counter++) + "_" + std::string(suffix));
}

inline BitMatrix make_sa_cluster_matrix() {
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

inline BitMatrix make_cuda_equivalence_matrix(std::size_t rows = 28, std::size_t cols = 96) {
    BitMatrix matrix(rows, cols);

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

inline int run_tests(const std::vector<TestCase>& tests) {
    for (const TestCase& test : tests) {
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "All tests passed\n";
    return 0;
}

} // namespace bmmpy::test
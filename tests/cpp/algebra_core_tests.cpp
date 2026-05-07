#include "bmmpy/algebra/mastrovito.hpp"
#include "bmmpy/algebra/transforms.hpp"
#include "bmmpy/core/bit_vector.hpp"
#include "test_common.hpp"

#include <initializer_list>
#include <vector>

using bmmpy::test::TestCase;

namespace {
using bmmpy::BitMatrix;
using bmmpy::BitVector;
using bmmpy::algebra::MastrovitoCore;

BitVector bits(std::size_t bit_count, std::initializer_list<std::size_t> set_bits) {
    return BitVector::from_positions(bit_count, std::vector<std::size_t>(set_bits));
}

void require_same_matrix(const BitMatrix& actual, const BitMatrix& expected, const char* context) {
    bmmpy::test::require(actual.rows() == expected.rows(), std::string(context) + ": row mismatch");
    bmmpy::test::require(actual.cols() == expected.cols(), std::string(context) + ": col mismatch");

    for (std::size_t row = 0; row < actual.rows(); ++row) {
        for (std::size_t col = 0; col < actual.cols(); ++col) {
            if (actual.get(row, col) != expected.get(row, col))
                bmmpy::test::fail(std::string(context) + ": matrix contents mismatch");
        }
    }
}

void test_bitmatrix_power_big_matches_u32() {
    const BitMatrix matrix = bmmpy::test::matrix_from_rows({
        "001",
        "101",
        "010",
    });

    const BitMatrix expected = matrix.power(5u);
    const BitMatrix actual = matrix.power(BitVector::from_u64(5));

    require_same_matrix(actual, expected, "bitmatrix_power_big_matches_u32");
}

void test_find_row_transform() {
    const BitMatrix source = bmmpy::test::matrix_from_rows({
        "1011",
        "0110",
        "1100",
    });

    const BitMatrix expected_transform = bmmpy::test::matrix_from_rows({
        "110",
        "011",
    });

    const BitMatrix target = expected_transform.mul(source);
    const BitMatrix actual_transform = bmmpy::algebra::find_row_transform(source, target);

    require_same_matrix(actual_transform.mul(source), target, "find_row_transform");
}

void test_find_row_transform_rejects_out_of_span() {
    const BitMatrix source = bmmpy::test::matrix_from_rows({
        "10",
        "10",
    });

    const BitMatrix target = bmmpy::test::matrix_from_rows({
        "01",
    });

    bmmpy::test::expect_runtime_error(
        [&]() { static_cast<void>(bmmpy::algebra::find_row_transform(source, target)); },
        "find_row_transform_rejects_out_of_span");
}

void test_mastrovito_blocks() {
    const MastrovitoCore mastrovito(3, {3, 1, 0}, bits(3, {1}));

    bmmpy::test::expect_rows(mastrovito.get_mastrovito_matrix(BitVector::from_u64(0)),
                             {"100", "010", "001"});

    bmmpy::test::expect_rows(mastrovito.get_mastrovito_matrix(BitVector::from_u64(1)),
                             {"001", "101", "010"});
}

void test_build_check_matrix() {
    const MastrovitoCore mastrovito(3, {3, 1, 0}, bits(3, {1}));

    bmmpy::test::expect_rows(mastrovito.build_check_matrix(2, 1, BitVector::from_u64(1)),
                             {"100001", "010101", "001010"});
}

void test_basis_change_matches_conjugation() {
    const std::vector<BitVector> basis = {
        bits(3, {0}),
        bits(3, {1}),
        bits(3, {2, 0}),
    };

    const MastrovitoCore standard(3, {3, 1, 0}, bits(3, {1}));
    const MastrovitoCore changed(3, {3, 1, 0}, bits(3, {1}), basis);

    const BitMatrix change = bmmpy::algebra::build_basis_change_matrix(basis, 3);
    const BitMatrix change_inv = bmmpy::algebra::invert_matrix(change);

    const BitMatrix expected =
        change_inv.mul(standard.get_mastrovito_matrix(BitVector::from_u64(1))).mul(change);

    require_same_matrix(changed.get_mastrovito_matrix(BitVector::from_u64(1)),
                        expected,
                        "basis_change_matches_conjugation");
}

void test_basis_coordinate_roundtrip() {
    const std::vector<BitVector> basis = {
        bits(3, {0}),
        bits(3, {1}),
        bits(3, {2, 0}),
    };

    const BitMatrix change = bmmpy::algebra::build_basis_change_matrix(basis, 3);
    const BitMatrix change_inv = bmmpy::algebra::invert_matrix(change);

    const BitVector mask = bits(3, {1, 2});
    const BitVector field = bmmpy::algebra::basis_mask_to_field_element(mask, basis, 3);
    const BitVector roundtrip = bmmpy::algebra::field_element_to_basis_mask(field, change_inv);

    for (std::size_t bit_index = 0; bit_index < mask.bit_count(); ++bit_index) {
        bmmpy::test::require(roundtrip.get(bit_index) == mask.get(bit_index),
                             "basis_coordinate_roundtrip");
    }
}

} // namespace

int main() {
    const std::vector<TestCase> tests = {
        {"bitmatrix_power_big_matches_u32", &test_bitmatrix_power_big_matches_u32},
        {"find_row_transform", &test_find_row_transform},
        {"find_row_transform_rejects_out_of_span", &test_find_row_transform_rejects_out_of_span},
        {"mastrovito_blocks", &test_mastrovito_blocks},
        {"build_check_matrix", &test_build_check_matrix},
        {"basis_change_matches_conjugation", &test_basis_change_matches_conjugation},
        {"basis_coordinate_roundtrip", &test_basis_coordinate_roundtrip},
    };

    return bmmpy::test::run_tests(tests);
}
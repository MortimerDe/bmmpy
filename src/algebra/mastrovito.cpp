#include "bmmpy/algebra/mastrovito.hpp"

#include "bmmpy/algebra/transforms.hpp"
#include "bmmpy/core/detail/bit_intrinsics.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace bmmpy::algebra {
namespace {

std::vector<std::size_t> normalize_poly_powers(std::vector<std::size_t> powers) {
    if (powers.empty())
        throw std::invalid_argument("polynomial must contain at least one term");

    std::sort(powers.begin(), powers.end());
    powers.erase(std::unique(powers.begin(), powers.end()), powers.end());
    std::reverse(powers.begin(), powers.end());
    return powers;
}

void validate_standard_bits(const BitVector& bits, std::size_t degree, const char* context) {
    if (bits.bit_count() != degree)
        throw std::invalid_argument(std::string(context) + ": bit width must match degree");
}

BitVector multiply_by_x(const BitVector& value,
                        std::size_t degree,
                        const std::vector<std::size_t>& poly_powers) {
    validate_standard_bits(value, degree, "multiply_by_x");

    BitVector result(degree);

    for (std::size_t i = degree; i > 1; --i) {
        if (value.get(i - 2))
            result.set(i - 1, true);
    }

    if (value.get(degree - 1)) {
        for (std::size_t power : poly_powers) {
            if (power < degree)
                result.flip(power);
        }
    }

    return result;
}

BitVector monomial_to_field_element(std::size_t power,
                                    std::size_t degree,
                                    const std::vector<std::size_t>& poly_powers) {
    if (degree == 0)
        throw std::invalid_argument("monomial_to_field_element: degree must be positive");

    BitVector value(degree);
    value.set(0, true);

    for (std::size_t i = 0; i < power; ++i)
        value = multiply_by_x(value, degree, poly_powers);

    return value;
}

BitMatrix build_x_matrix(std::size_t degree, const std::vector<std::size_t>& poly_powers) {
    BitMatrix result(degree, degree);
    std::vector<bool> reduction_coeffs(degree, false);

    for (std::size_t power : poly_powers) {
        if (power < degree)
            reduction_coeffs[power] = true;
    }

    for (std::size_t row = 0; row < degree; ++row) {
        if (row > 0)
            result.set(row, row - 1, true);

        if (reduction_coeffs[row])
            result.set(row, degree - 1, true);
    }

    return result;
}

std::vector<BitMatrix> build_standard_power_matrices(std::size_t degree,
                                                     const std::vector<std::size_t>& poly_powers) {
    std::vector<BitMatrix> power_matrices;
    power_matrices.reserve(degree);

    const BitMatrix mx = build_x_matrix(degree, poly_powers);
    power_matrices.push_back(BitMatrix::identity(degree));

    for (std::size_t i = 1; i < degree; ++i)
        power_matrices.push_back(power_matrices.back().mul(mx));

    return power_matrices;
}

BitMatrix combine_power_matrices(const std::vector<BitMatrix>& power_matrices,
                                 const BitVector& coeff_bits,
                                 std::size_t degree) {
    BitMatrix result(degree, degree);

    coeff_bits.for_each_set_bit([&](std::size_t bit_index) {
        if (bit_index >= power_matrices.size())
            throw std::invalid_argument(
                "combine_power_matrices: coefficient bit exceeds basis size");
        result.xor_assign(power_matrices[bit_index]);
    });

    return result;
}

void write_block(BitMatrix& out,
                 std::size_t row_offset,
                 std::size_t col_offset,
                 const BitMatrix& block) {
    for (std::size_t row = 0; row < block.rows(); ++row) {
        out.row_or_words(row_offset + row, col_offset, block.row_words(row), block.cols());
    }
}

} // namespace

BitVector powers_to_field_element(const std::vector<std::size_t>& powers,
                                  std::size_t degree,
                                  const std::vector<std::size_t>& poly_powers) {
    BitVector value(degree);

    for (std::size_t power : powers)
        value.xor_assign(monomial_to_field_element(power, degree, poly_powers));

    return value;
}

BitMatrix build_basis_change_matrix(const std::vector<BitVector>& basis_elements,
                                    std::size_t degree) {
    if (basis_elements.size() != degree)
        throw std::invalid_argument("basis size must equal degree");

    BitMatrix matrix(degree, degree);

    for (std::size_t col = 0; col < degree; ++col) {
        validate_standard_bits(basis_elements[col], degree, "basis element");

        basis_elements[col].for_each_set_bit([&](std::size_t row) { matrix.set(row, col, true); });
    }

    if (matrix.rank() != degree)
        throw std::invalid_argument("basis elements must be linearly independent");

    return matrix;
}

BitVector basis_mask_to_field_element(const BitVector& mask,
                                      const std::vector<BitVector>& basis_elements,
                                      std::size_t degree) {
    if (mask.bit_count() != degree)
        throw std::invalid_argument("basis mask width must equal degree");

    if (basis_elements.empty())
        return mask;

    if (basis_elements.size() != degree)
        throw std::invalid_argument("basis size must equal degree");

    BitVector value(degree);

    mask.for_each_set_bit(
        [&](std::size_t bit_index) { value.xor_assign(basis_elements[bit_index]); });

    return value;
}

BitVector field_element_to_basis_mask(const BitVector& element_bits, const BitMatrix& change_inv) {
    if (change_inv.rows() != change_inv.cols())
        throw std::invalid_argument("change_inv must be square");
    if (element_bits.bit_count() != change_inv.cols())
        throw std::invalid_argument("element width must match change_inv");

    BitVector mask(change_inv.rows());

    for (std::size_t row = 0; row < change_inv.rows(); ++row) {
        const std::uint64_t* lhs = change_inv.row_words(row);
        bool parity = false;

        for (std::size_t word_index = 0; word_index < change_inv.words_per_row(); ++word_index)
            parity ^= bmmpy::detail::parity64(lhs[word_index] & element_bits.words()[word_index]);

        if (parity)
            mask.set(row, true);
    }

    return mask;
}

MastrovitoCore::MastrovitoCore(std::size_t degree,
                               std::vector<std::size_t> poly_powers,
                               BitVector element_standard_bits,
                               std::vector<BitVector> basis_elements)
    : _degree(degree), _poly_powers(normalize_poly_powers(std::move(poly_powers))),
      _element_standard_bits(std::move(element_standard_bits)),
      _basis_elements(std::move(basis_elements)) {
    if (_degree == 0)
        throw std::invalid_argument("degree must be positive");
    if (_poly_powers.front() != _degree)
        throw std::invalid_argument("polynomial degree must match highest power");

    validate_standard_bits(_element_standard_bits, _degree, "element");
    if (_element_standard_bits.none())
        throw std::invalid_argument("element must be non-zero");

    const std::vector<BitMatrix> standard_power_matrices =
        build_standard_power_matrices(_degree, _poly_powers);

    if (_basis_elements.empty()) {
        _power_basis_matrices = standard_power_matrices;
    } else {
        const BitMatrix change = build_basis_change_matrix(_basis_elements, _degree);
        const BitMatrix change_inv = invert_matrix(change);

        _power_basis_matrices.reserve(standard_power_matrices.size());
        for (const BitMatrix& matrix : standard_power_matrices)
            _power_basis_matrices.push_back(change_inv.mul(matrix).mul(change));
    }

    _m_alpha = combine_power_matrices(_power_basis_matrices, _element_standard_bits, _degree);
}

std::vector<BitMatrix> MastrovitoCore::get_basis_multiplication_matrices() const {
    return _power_basis_matrices;
}

BitMatrix MastrovitoCore::get_mastrovito_matrix(const BitVector& power) const {
    return _m_alpha.power(power);
}

BitMatrix MastrovitoCore::build_check_matrix(std::size_t c, std::size_t k) const {
    return build_check_matrix(c, k, BitVector::from_u64(0));
}

BitMatrix
MastrovitoCore::build_check_matrix(std::size_t c, std::size_t k, const BitVector& start_i) const {
    if (c < k)
        throw std::invalid_argument("c must be greater than or equal to k");

    const std::size_t row_block_count = c - k;
    BitMatrix out(row_block_count * _degree, c * _degree);

    if (row_block_count == 0 || c == 0)
        return out;

    BitMatrix row_step = _m_alpha.power(start_i);

    for (std::size_t row_block = 0; row_block < row_block_count; ++row_block) {
        BitMatrix current = BitMatrix::identity(_degree);

        for (std::size_t col_block = 0; col_block < c; ++col_block) {
            write_block(out, row_block * _degree, col_block * _degree, current);

            if (col_block + 1 < c)
                current = current.mul(row_step);
        }

        row_step = row_step.mul(_m_alpha);
    }

    return out;
}

} // namespace bmmpy::algebra
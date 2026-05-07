#include "bmmpy/algebra/transforms.hpp"

#include "bmmpy/core/bit_vector.hpp"

#include <utility>
#include <vector>

namespace bmmpy::algebra {
namespace {

struct BasisEntry {
    std::size_t pivot_col;
    BitVector row_bits;
    BitVector coeff_bits;
};

std::vector<BasisEntry> build_row_basis(const BitMatrix& source) {
    std::vector<BasisEntry> basis;

    for (std::size_t row = 0; row < source.rows(); ++row) {
        BitVector reduced = source.row_bits(row);
        BitVector coeff = BitVector::unit(source.rows(), row);

        for (const BasisEntry& entry : basis) {
            if (reduced.get(entry.pivot_col)) {
                reduced.xor_assign(entry.row_bits);
                coeff.xor_assign(entry.coeff_bits);
            }
        }

        if (reduced.none())
            continue;

        BasisEntry entry{
            reduced.highest_set_bit(),
            std::move(reduced),
            std::move(coeff),
        };

        auto it = basis.begin();
        while (it != basis.end() && it->pivot_col > entry.pivot_col)
            ++it;

        basis.insert(it, std::move(entry));
    }

    return basis;
}

BitVector solve_in_basis(BitVector reduced,
                         const std::vector<BasisEntry>& basis,
                         std::size_t source_row_count) {
    BitVector coeff(source_row_count);

    for (const BasisEntry& entry : basis) {
        if (reduced.get(entry.pivot_col)) {
            reduced.xor_assign(entry.row_bits);
            coeff.xor_assign(entry.coeff_bits);
        }
    }

    if (reduced.any())
        throw RowTransformError("target contains a row outside the row span of source");

    return coeff;
}

} // namespace

BitMatrix find_row_transform(const BitMatrix& source, const BitMatrix& target) {
    if (source.cols() != target.cols())
        throw std::invalid_argument("source and target must have the same number of columns");

    const std::vector<BasisEntry> basis = build_row_basis(source);
    BitMatrix transform(target.rows(), source.rows());

    for (std::size_t row = 0; row < target.rows(); ++row) {
        BitVector coeff = solve_in_basis(target.row_bits(row), basis, source.rows());
        transform.set_row_bits(row, coeff);
    }

    return transform;
}

BitMatrix invert_matrix(const BitMatrix& matrix) {
    if (matrix.rows() != matrix.cols())
        throw std::invalid_argument("invert_matrix: matrix must be square");

    BitMatrix work(matrix);
    BitMatrix inverse = BitMatrix::identity(matrix.rows());

    for (std::size_t col = 0; col < matrix.cols(); ++col) {
        std::size_t pivot = col;
        while (pivot < matrix.rows() && !work.get_unchecked(pivot, col))
            ++pivot;

        if (pivot == matrix.rows())
            throw std::invalid_argument("invert_matrix: matrix is singular");

        if (pivot != col) {
            work.swap_rows(col, pivot);
            inverse.swap_rows(col, pivot);
        }

        for (std::size_t row = 0; row < matrix.rows(); ++row) {
            if (row != col && work.get_unchecked(row, col)) {
                work.row_xor(row, col);
                inverse.row_xor(row, col);
            }
        }
    }

    return inverse;
}

} // namespace bmmpy::algebra
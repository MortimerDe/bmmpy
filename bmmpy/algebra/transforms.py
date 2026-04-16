"""
Linear row-space transformation helpers over GF(2).
"""

from __future__ import annotations

from ..matrix import BitMatrix


class RowTransformError(ValueError):
    """Raised when a target matrix is not in the row span of a source matrix."""


def find_row_transform(source: BitMatrix, target: BitMatrix) -> BitMatrix:
    """Return T such that T @ source == target.

    Parameters
    ----------
    source : BitMatrix
        Source matrix A.
    target : BitMatrix
        Target matrix B.

    Returns
    -------
    BitMatrix
        Transformation matrix T with shape (target.rows, source.rows).

    Raises
    ------
    ValueError
        If source and target have different column counts.
    RowTransformError
        If a target row is not representable as an XOR combination of source rows.

    Notes
    -----
    The returned matrix is a row-space transform over GF(2). It is not required
    to be square or invertible. The contract is simply ``T @ source == target``.
    """
    if source.cols != target.cols:
        raise ValueError("source and target must have the same number of columns")

    basis = _build_row_basis(source)
    transform = BitMatrix(target.rows, source.rows)

    for target_row_index, target_bits in enumerate(_matrix_rows_as_ints(target)):
        coeff_bits = _solve_in_basis(target_bits, basis)
        _write_mask_row(transform, target_row_index, coeff_bits)

    return transform


def _build_row_basis(source: BitMatrix) -> list[tuple[int, int, int]]:
    """Build a descending-pivot row basis.

    Each entry is ``(pivot_col, row_bits, coeff_bits)``, where ``coeff_bits``
    encodes how the basis row is formed from the original source rows.
    """
    basis: list[tuple[int, int, int]] = []

    for row_index, row_bits in enumerate(_matrix_rows_as_ints(source)):
        reduced = row_bits
        coeff_bits = 1 << row_index

        for pivot_col, basis_row_bits, basis_coeff_bits in basis:
            if (reduced >> pivot_col) & 1:
                reduced ^= basis_row_bits
                coeff_bits ^= basis_coeff_bits

        if reduced == 0:
            continue

        pivot_col = reduced.bit_length() - 1
        entry = (pivot_col, reduced, coeff_bits)

        inserted = False
        for insert_index, (existing_pivot, _, _) in enumerate(basis):
            if pivot_col > existing_pivot:
                basis.insert(insert_index, entry)
                inserted = True
                break

        if not inserted:
            basis.append(entry)

    return basis


def _solve_in_basis(row_bits: int, basis: list[tuple[int, int, int]]) -> int:
    reduced = row_bits
    coeff_bits = 0

    for pivot_col, basis_row_bits, basis_coeff_bits in basis:
        if (reduced >> pivot_col) & 1:
            reduced ^= basis_row_bits
            coeff_bits ^= basis_coeff_bits

    if reduced != 0:
        raise RowTransformError("target contains a row outside the row span of source")

    return coeff_bits


def _matrix_rows_as_ints(matrix: BitMatrix) -> list[int]:
    rows = matrix.to_rows()
    return [int(row[::-1], 2) if row else 0 for row in rows]


def _write_mask_row(matrix: BitMatrix, row_index: int, mask_bits: int) -> None:
    bits = mask_bits
    while bits:
        low_bit = bits & -bits
        bit_index = low_bit.bit_length() - 1
        matrix[row_index, bit_index] = True
        bits ^= low_bit


__all__ = ["RowTransformError", "find_row_transform"]
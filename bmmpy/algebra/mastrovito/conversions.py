from __future__ import annotations

from collections.abc import Sequence

from ...matrix import BitMatrix
from ..transforms import find_row_transform
from .bit_algebra import powers_to_field_element
from .parsing import (
    BasisElementLike,
    BasisLike,
    FieldElementLike,
    parse_field_element,
)


def rows_to_matrix(rows: Sequence[int], row_count: int, col_count: int) -> BitMatrix:
    matrix = BitMatrix(row_count, col_count)

    for row_index, row_bits in enumerate(rows):
        bits = row_bits
        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            matrix[row_index, bit_index] = True
            bits ^= low_bit

    return matrix


def matrix_to_rows(matrix: BitMatrix) -> list[int]:
    return [int(row[::-1], 2) if row else 0 for row in matrix.to_rows()]


def write_block(
    matrix: BitMatrix,
    row_offset: int,
    col_offset: int,
    rows: Sequence[int],
) -> None:
    for local_row, row_bits in enumerate(rows):
        bits = row_bits
        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            matrix[row_offset + local_row, col_offset + bit_index] = True
            bits ^= low_bit


def field_element_to_int(
    element: FieldElementLike,
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    powers = parse_field_element(element)
    return powers_to_field_element(powers, degree, poly_powers)


def basis_element_to_int(
    element: BasisElementLike,
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    if isinstance(element, int):
        if element < 0:
            raise ValueError("Basis monomial powers must be non-negative")
        return powers_to_field_element([element], degree, poly_powers)

    if isinstance(element, str):
        return field_element_to_int(element, degree, poly_powers)

    if isinstance(element, Sequence) and not isinstance(element, (bytes, bytearray, str)):
        return field_element_to_int(element, degree, poly_powers)

    raise TypeError(
        "basis elements must be ints, polynomial strings, or sequences of powers"
    )


def build_basis_change_matrix(
    basis: BasisLike,
    degree: int,
    poly_powers: Sequence[int],
) -> BitMatrix:
    if isinstance(basis, (str, bytes, bytearray)):
        raise TypeError("basis must be a sequence of basis elements, not a single string")
    if len(basis) != degree:
        raise ValueError(f"basis must contain exactly {degree} elements")

    matrix = BitMatrix(degree, degree)

    for col_index, element in enumerate(basis):
        element_bits = basis_element_to_int(element, degree, poly_powers)
        for row_index in range(degree):
            if (element_bits >> row_index) & 1:
                matrix[row_index, col_index] = True

    if matrix.rank() != degree:
        raise ValueError("basis elements must be linearly independent")

    return matrix


def invert_matrix(matrix: BitMatrix) -> BitMatrix:
    return find_row_transform(matrix, BitMatrix.identity(matrix.rows))


__all__ = [
    "basis_element_to_int",
    "build_basis_change_matrix",
    "field_element_to_int",
    "invert_matrix",
    "matrix_to_rows",
    "rows_to_matrix",
    "write_block",
]
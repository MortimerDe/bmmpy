from __future__ import annotations

from collections.abc import Sequence

from ...matrix import BitMatrix
from ..._bmmpy import (
    BitVector as _NativeBitVector,
    basis_mask_to_field_element as _native_basis_mask_to_field_element,
    build_basis_change_matrix as _native_build_basis_change_matrix,
    field_element_to_basis_mask as _native_field_element_to_basis_mask,
    invert_matrix as _native_invert_matrix,
    powers_to_field_element as _native_powers_to_field_element,
)
from .parsing import BasisElementLike, BasisLike, FieldElementLike, parse_field_element


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


def write_block(matrix: BitMatrix, row_offset: int, col_offset: int, rows: Sequence[int]) -> None:
    for local_row, row_bits in enumerate(rows):
        bits = row_bits
        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            matrix[row_offset + local_row, col_offset + bit_index] = True
            bits ^= low_bit


def _int_to_words(value: int) -> list[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an int")
    if value < 0:
        raise ValueError("value must be non-negative")

    words: list[int] = []
    while value:
        words.append(value & ((1 << 64) - 1))
        value >>= 64

    return words or [0]


def _int_to_bitvector(value: int, bit_count: int | None = None) -> _NativeBitVector:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an int")
    if value < 0:
        raise ValueError("value must be non-negative")

    if bit_count is None:
        bit_count = max(1, value.bit_length())

    return _NativeBitVector.from_words(bit_count, _int_to_words(value))


def _bitvector_to_int(bits: _NativeBitVector) -> int:
    value = 0
    for word_index, word in enumerate(bits.to_words()):
        value |= int(word) << (64 * word_index)
    return value


def field_element_to_int(
    element: FieldElementLike,
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    powers = parse_field_element(element)
    return _bitvector_to_int(_native_powers_to_field_element(list(powers), degree, list(poly_powers)))


def basis_element_to_int(
    element: BasisElementLike,
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    if isinstance(element, int):
        if element < 0:
            raise ValueError("Basis monomial powers must be non-negative")
        return _bitvector_to_int(
            _native_powers_to_field_element([element], degree, list(poly_powers))
        )

    return field_element_to_int(element, degree, poly_powers)


def _basis_to_bitvectors(
    basis: BasisLike,
    degree: int,
    poly_powers: Sequence[int],
) -> list[_NativeBitVector]:
    return [
        _int_to_bitvector(basis_element_to_int(element, degree, poly_powers), degree)
        for element in basis
    ]


def build_basis_change_matrix(
    basis: BasisLike,
    degree: int,
    poly_powers: Sequence[int],
) -> BitMatrix:
    if isinstance(basis, (str, bytes, bytearray)):
        raise TypeError("basis must be a sequence of basis elements, not a single string")
    if len(basis) != degree:
        raise ValueError(f"basis must contain exactly {degree} elements")

    return _native_build_basis_change_matrix(
        _basis_to_bitvectors(basis, degree, poly_powers),
        degree,
    )


def invert_matrix(matrix: BitMatrix) -> BitMatrix:
    return _native_invert_matrix(matrix)


def basis_mask_to_field_element(
    mask: int,
    basis: BasisLike | None,
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    if isinstance(mask, bool) or not isinstance(mask, int):
        raise TypeError("element_mask must be an int")
    if mask < 0:
        raise ValueError("element_mask must be non-negative")
    if mask >> degree:
        raise ValueError("element_mask requires more basis coordinates than available")

    if basis is None:
        return mask

    basis_bits = _basis_to_bitvectors(basis, degree, poly_powers)
    return _bitvector_to_int(
        _native_basis_mask_to_field_element(_int_to_bitvector(mask, degree), basis_bits, degree)
    )


def field_element_to_basis_mask(
    element_bits: int,
    basis: BasisLike | None,
    degree: int,
    poly_powers: Sequence[int],
    *,
    change_inv: BitMatrix | None = None,
) -> int:
    if isinstance(element_bits, bool) or not isinstance(element_bits, int):
        raise TypeError("element_bits must be an int")
    if element_bits < 0:
        raise ValueError("element_bits must be non-negative")
    if element_bits >> degree:
        raise ValueError("element_bits requires more standard basis coordinates than available")

    if basis is None:
        return element_bits

    if change_inv is None:
        change_inv = invert_matrix(build_basis_change_matrix(basis, degree, poly_powers))

    return _bitvector_to_int(
        _native_field_element_to_basis_mask(
            _int_to_bitvector(element_bits, degree),
            change_inv,
        )
    )
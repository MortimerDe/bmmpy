"""
Mastrovito-based generators for binary parity-check matrices.

This module provides a pure Python baseline implementation that builds native
BitMatrix objects using the existing bmmpy matrix container.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from .matrix import BitMatrix

PolynomialLike: TypeAlias = str | tuple[int, Sequence[int]] | Sequence[int]


def parse_poly(poly: PolynomialLike) -> tuple[int, list[int]]:
    """Parse or normalize a polynomial representation.

    Parameters
    ----------
    poly : str | tuple[int, Sequence[int]] | Sequence[int]
        Polynomial representation. Supported forms are:
        - string, for example ``"x^8 + x^5 + x^3 + x + 1"``
        - tuple, for example ``(8, [8, 5, 3, 1, 0])``
        - sequence of powers, for example ``[8, 5, 3, 1, 0]``

    Returns
    -------
    tuple[int, list[int]]
        Normalized ``(degree, powers)`` pair with powers sorted in descending
        order and duplicate powers removed.
    """
    if isinstance(poly, str):
        return _parse_poly_string(poly)

    if isinstance(poly, tuple):
        if len(poly) != 2:
            raise TypeError("Polynomial tuple must have the form (degree, powers)")

        degree, powers = poly
        if not isinstance(degree, int):
            raise TypeError("Polynomial degree must be an int")

        normalized_degree, normalized_powers = _normalize_powers(powers)
        if degree != normalized_degree:
            raise ValueError(
                f"Polynomial degree {degree} does not match the highest power "
                f"{normalized_degree}"
            )

        return normalized_degree, normalized_powers

    if isinstance(poly, Sequence) and not isinstance(poly, (bytes, bytearray)):
        return _normalize_powers(poly)

    raise TypeError("Unsupported polynomial format. Expected str, tuple, or sequence of ints.")


class Mastrovito:
    """Build Mastrovito matrices and parity-check matrices for a fixed polynomial.

    Parameters
    ----------
    poly : str | tuple[int, Sequence[int]] | Sequence[int]
        Irreducible polynomial representation.
    elem : int, default=2
        Base field element used to build the multiplication matrix. The default
        value ``2`` corresponds to ``x`` in the polynomial basis.
    """

    __slots__ = ("degree", "powers", "elem", "_m_alpha", "_period")

    def __init__(self, poly: PolynomialLike, *, elem: int = 2) -> None:
        degree, powers = parse_poly(poly)

        if degree <= 0:
            raise ValueError("Polynomial degree must be positive")
        if elem <= 0:
            raise ValueError("Element must be positive")
        if elem >= (1 << degree):
            raise ValueError(
                f"Element {elem} does not fit into the degree-{degree} field representation"
            )

        self.degree = degree
        self.powers = powers
        self.elem = elem
        self._period = (1 << degree) - 1
        self._m_alpha = _build_element_matrix(degree, powers, elem)

    def __repr__(self) -> str:
        return (
            "MastrovitoGenerator("
            f"degree={self.degree}, powers={self.powers}, elem={self.elem})"
        )

    def get_mastrovito_matrix(self, power: int) -> BitMatrix:
        """Return the degree x degree Mastrovito block for the given power."""
        if not isinstance(power, int):
            raise TypeError("power must be an int")

        rows = _matrix_power_rows(self._m_alpha, power % self._period, self.degree)
        return _rows_to_matrix(rows, self.degree, self.degree)

    def build_check_matrix(self, c: int, k: int, *, start_i: int = 0) -> BitMatrix:
        """Build a parity-check matrix using Mastrovito blocks.

        Parameters
        ----------
        c : int
            Total number of block columns.
        k : int
            Number of information block columns.
        start_i : int, default=0
            Starting step index for the block-row sequence.
        """
        if not isinstance(c, int):
            raise TypeError("c must be an int")
        if not isinstance(k, int):
            raise TypeError("k must be an int")
        if not isinstance(start_i, int):
            raise TypeError("start_i must be an int")
        if c < 0 or k < 0:
            raise ValueError("c and k must be non-negative")
        if c < k:
            raise ValueError("c must be greater than or equal to k")

        row_block_count = c - k
        rows = row_block_count * self.degree
        cols = c * self.degree
        matrix = BitMatrix(rows, cols)

        if rows == 0 or cols == 0:
            return matrix

        required_powers = {
            ((start_i + row_block) * col_block) % self._period
            for row_block in range(row_block_count)
            for col_block in range(c)
        }
        blocks = _precompute_blocks(self._m_alpha, self.degree, required_powers)

        for row_block in range(row_block_count):
            row_offset = row_block * self.degree
            step = start_i + row_block

            for col_block in range(c):
                col_offset = col_block * self.degree
                power = (step * col_block) % self._period
                _write_block(matrix, row_offset, col_offset, blocks[power])

        return matrix


def build_check_matrix(
    poly: PolynomialLike,
    c: int,
    k: int,
    *,
    start_i: int = 0,
    elem: int = 2,
) -> BitMatrix:
    """Convenience wrapper for MastrovitoGenerator.build_check_matrix()."""
    generator = MastrovitoGenerator(poly, elem=elem)
    return generator.build_check_matrix(c, k, start_i=start_i)


def get_mastrovito_matrix(
    poly: PolynomialLike,
    power: int,
    *,
    elem: int = 2,
) -> BitMatrix:
    """Convenience wrapper for MastrovitoGenerator.get_mastrovito_matrix()."""
    generator = MastrovitoGenerator(poly, elem=elem)
    return generator.get_mastrovito_matrix(power)


def _parse_poly_string(poly: str) -> tuple[int, list[int]]:
    compact = poly.replace(" ", "")
    if not compact:
        raise ValueError("Polynomial string is empty")

    terms = [term for term in compact.replace("-", "+").split("+") if term]
    if not terms:
        raise ValueError("Polynomial string is empty")

    powers: list[int] = []
    for term in terms:
        if term == "1":
            powers.append(0)
            continue

        if term == "x":
            powers.append(1)
            continue

        if term.startswith("x**"):
            exponent_text = term[3:]
        elif term.startswith("x^"):
            exponent_text = term[2:]
        else:
            raise ValueError(f"Unsupported polynomial term: {term!r}")

        if not exponent_text.isdigit():
            raise ValueError(f"Invalid exponent in polynomial term: {term!r}")

        powers.append(int(exponent_text))

    return _normalize_powers(powers)


def _normalize_powers(values: Sequence[int]) -> tuple[int, list[int]]:
    powers: list[int] = []

    for value in values:
        if not isinstance(value, int):
            raise TypeError("Polynomial powers must be integers")
        if value < 0:
            raise ValueError("Polynomial powers must be non-negative")
        powers.append(value)

    if not powers:
        raise ValueError("Polynomial must contain at least one term")

    normalized = sorted(set(powers), reverse=True)
    return normalized[0], normalized


def _identity_rows(degree: int) -> list[int]:
    return [1 << index for index in range(degree)]


def _build_x_matrix(degree: int, powers: Sequence[int]) -> list[int]:
    coeffs = {power for power in powers if power < degree}
    rows = [0] * degree

    for row_index in range(degree):
        row_bits = 0

        if row_index > 0:
            row_bits |= 1 << (row_index - 1)

        if row_index in coeffs:
            row_bits |= 1 << (degree - 1)

        rows[row_index] = row_bits

    return rows


def _build_element_matrix(degree: int, powers: Sequence[int], elem: int) -> list[int]:
    mx = _build_x_matrix(degree, powers)
    mx_powers = [_identity_rows(degree)]

    for _ in range(1, degree):
        mx_powers.append(_mul_matrix_rows(mx_powers[-1], mx, degree))

    rows = [0] * degree
    for bit_index in range(degree):
        if (elem >> bit_index) & 1:
            source = mx_powers[bit_index]
            for row_index, row_bits in enumerate(source):
                rows[row_index] ^= row_bits

    return rows


def _mul_matrix_rows(left: Sequence[int], right: Sequence[int], degree: int) -> list[int]:
    result = [0] * degree

    for row_index, row_bits in enumerate(left):
        value = 0
        bits = row_bits

        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            value ^= right[bit_index]
            bits ^= low_bit

        result[row_index] = value

    return result


def _matrix_power_rows(base: Sequence[int], power: int, degree: int) -> list[int]:
    result = _identity_rows(degree)
    factor = list(base)
    exponent = power

    while exponent:
        if exponent & 1:
            result = _mul_matrix_rows(result, factor, degree)

        exponent >>= 1
        if exponent:
            factor = _mul_matrix_rows(factor, factor, degree)

    return result


def _precompute_blocks(
    base: Sequence[int],
    degree: int,
    powers: set[int],
) -> dict[int, list[int]]:
    targets = sorted(powers)
    blocks: dict[int, list[int]] = {}

    current_power = 0
    current_rows = _identity_rows(degree)

    for target in targets:
        while current_power < target:
            current_rows = _mul_matrix_rows(current_rows, base, degree)
            current_power += 1

        blocks[target] = current_rows.copy()

    return blocks


def _rows_to_matrix(rows: Sequence[int], row_count: int, col_count: int) -> BitMatrix:
    matrix = BitMatrix(row_count, col_count)

    for row_index, row_bits in enumerate(rows):
        bits = row_bits
        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            matrix[row_index, bit_index] = True
            bits ^= low_bit

    return matrix


def _write_block(matrix: BitMatrix, row_offset: int, col_offset: int, rows: Sequence[int]) -> None:
    for local_row, row_bits in enumerate(rows):
        bits = row_bits
        while bits:
            low_bit = bits & -bits
            bit_index = low_bit.bit_length() - 1
            matrix[row_offset + local_row, col_offset + bit_index] = True
            bits ^= low_bit


__all__ = [
    "MastrovitoGenerator",
    "build_check_matrix",
    "get_mastrovito_matrix",
    "parse_poly",
]
"""Public Mastrovito generator and convenience helpers."""

from __future__ import annotations

from ...matrix import BitMatrix
from .bit_algebra import (
    build_element_matrix,
    matrix_power_rows,
    precompute_blocks,
)
from .conversions import (
    build_basis_change_matrix,
    field_element_to_int,
    invert_matrix,
    matrix_to_rows,
    rows_to_matrix,
    write_block,
)
from .parsing import BasisLike, FieldElementLike, PolynomialLike, parse_poly


class Mastrovito:
    """Build Mastrovito matrices and parity-check matrices for a fixed polynomial.

    Parameters
    ----------
    poly : str | tuple[int, Sequence[int]] | Sequence[int]
        Irreducible polynomial representation.
    basis : Sequence[int | str | Sequence[int]] | None, default=None
        Ordered basis used for the returned coordinate system. Integers denote
        monomials ``x^i``. Strings and sequences denote general field elements
        in polynomial form.
    element : str | Sequence[int], default="x"
        Non-zero field element used to build the multiplication matrix.
        Examples: ``"x"``, ``"x^7 + x + 1"``, ``[7, 1, 0]``.
    """

    __slots__ = ("degree", "powers", "element", "_m_alpha", "_period")

    def __init__(
        self,
        poly: PolynomialLike,
        *,
        basis: BasisLike | None = None,
        element: FieldElementLike = "x",
    ) -> None:
        degree, powers = parse_poly(poly)

        if degree <= 0:
            raise ValueError("Polynomial degree must be positive")

        element_bits = field_element_to_int(element, degree, powers)
        if element_bits == 0:
            raise ValueError("element must be non-zero")

        self.degree = degree
        self.powers = powers
        self.element = element
        self._period = (1 << degree) - 1
        self._m_alpha = build_element_matrix(degree, powers, element_bits)

        if basis is not None:
            change = build_basis_change_matrix(basis, degree, powers)
            change_inv = invert_matrix(change)
            self._m_alpha = matrix_to_rows(
                change_inv @ rows_to_matrix(self._m_alpha, degree, degree) @ change
            )

    def __repr__(self) -> str:
        return (
            f"Mastrovito(degree={self.degree}, powers={self.powers}, "
            f"element={self.element!r})"
        )

    def get_mastrovito_matrix(self, power: int) -> BitMatrix:
        """Return the degree x degree Mastrovito block for the given power."""
        if not isinstance(power, int):
            raise TypeError("power must be an int")

        rows = matrix_power_rows(self._m_alpha, power % self._period, self.degree)
        return rows_to_matrix(rows, self.degree, self.degree)

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
        blocks = precompute_blocks(self._m_alpha, self.degree, required_powers)

        for row_block in range(row_block_count):
            row_offset = row_block * self.degree
            step = start_i + row_block

            for col_block in range(c):
                col_offset = col_block * self.degree
                power = (step * col_block) % self._period
                write_block(matrix, row_offset, col_offset, blocks[power])

        return matrix


def build_check_matrix(
    poly: PolynomialLike,
    c: int,
    k: int,
    *,
    start_i: int = 0,
    basis: BasisLike | None = None,
    element: FieldElementLike = "x",
) -> BitMatrix:
    generator = Mastrovito(poly, basis=basis, element=element)
    return generator.build_check_matrix(c, k, start_i=start_i)


def get_mastrovito_matrix(
    poly: PolynomialLike,
    power: int,
    *,
    basis: BasisLike | None = None,
    element: FieldElementLike = "x",
) -> BitMatrix:
    generator = Mastrovito(poly, basis=basis, element=element)
    return generator.get_mastrovito_matrix(power)


__all__ = [
    "Mastrovito",
    "build_check_matrix",
    "get_mastrovito_matrix",
]
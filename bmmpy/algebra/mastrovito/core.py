"""Public Mastrovito helpers built from the regular representation of GF(2^n).

Let E = (1, x, ..., x^(n-1)) be the standard polynomial basis and let
B = (b_0, ..., b_{n-1}) be a user-selected coordinate basis.

This module follows the construction

    rho_E(a) = sum_i a_i rho_E(x^i),
    rho_B(a) = S^{-1} rho_E(a) S,

where S = [b_0 ... b_{n-1}]_E is the change-of-basis matrix.

The implementation is intentionally structured in the same order:

1. build rho_E(1), rho_E(x), ..., rho_E(x^(n-1)),
2. conjugate that family by S when a custom basis is requested,
3. assemble rho_B(element) as a GF(2)-linear combination of the transformed
   family,
4. use powers of rho_B(element) as Mastrovito blocks.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...matrix import BitMatrix
from .bit_algebra import (
    build_standard_power_matrices,
    combine_power_matrices,
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


def _conjugate_rows(
    rows: Sequence[int],
    change: BitMatrix,
    change_inv: BitMatrix,
    degree: int,
) -> list[int]:
    return matrix_to_rows(change_inv @ rows_to_matrix(rows, degree, degree) @ change)


class Mastrovito:
    """Build Mastrovito blocks and parity-check matrices for a fixed polynomial.

    Parameters
    ----------
    poly : str | tuple[int, Sequence[int]] | Sequence[int]
        Irreducible polynomial representation.
    basis : Sequence[int | str | Sequence[int]] | None, default=None
        Ordered coordinate basis for the returned matrices. Integers denote
        monomials ``x^i``. Strings and sequences denote general field elements
        in polynomial form.
    element : str | Sequence[int], default="x"
        Non-zero field element whose multiplication operator defines the block
        generator. Examples: ``"x"``, ``"x^7 + x + 1"``, ``[7, 1, 0]``.

    Notes
    -----
    Field elements are still parsed as polynomials in ``x``. The ``basis``
    parameter only changes the coordinate system in which the returned matrices
    are written.
    """

    __slots__ = (
        "degree",
        "powers",
        "basis",
        "element",
        "_period",
        "_power_basis_rows",
        "_m_alpha",
    )

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
        self.basis = basis
        self.element = element
        self._period = (1 << degree) - 1

        standard_power_rows = build_standard_power_matrices(degree, powers)

        if basis is not None:
            change = build_basis_change_matrix(basis, degree, powers)
            change_inv = invert_matrix(change)
            power_basis_rows = [
                _conjugate_rows(rows, change, change_inv, degree)
                for rows in standard_power_rows
            ]
        else:
            power_basis_rows = [list(rows) for rows in standard_power_rows]

        self._power_basis_rows = tuple(tuple(rows) for rows in power_basis_rows)

        self._m_alpha = combine_power_matrices(
            self._power_basis_rows,
            element_bits,
            degree,
        )

    def __repr__(self) -> str:
        return (
            f"Mastrovito(degree={self.degree}, powers={self.powers}, "
            f"basis={self.basis!r}, element={self.element!r})"
        )

    def get_basis_multiplication_matrices(self) -> list[BitMatrix]:
        """Return [rho_B(1), rho_B(x), ..., rho_B(x^(n-1))]."""
        return [
            rows_to_matrix(rows, self.degree, self.degree)
            for rows in self._power_basis_rows
        ]

    def get_mastrovito_matrix(self, power: int) -> BitMatrix:
        """Return rho_B(element^power) as a degree x degree matrix."""
        if not isinstance(power, int):
            raise TypeError("power must be an int")

        rows = matrix_power_rows(self._m_alpha, power % self._period, self.degree)
        return rows_to_matrix(rows, self.degree, self.degree)

    def build_check_matrix(self, c: int, k: int, *, start_i: int = 0) -> BitMatrix:
        """Build a parity-check matrix with blocks rho_B(element^(step * j)).

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
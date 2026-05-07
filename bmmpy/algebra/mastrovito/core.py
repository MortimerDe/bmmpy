"""Public Mastrovito helpers built from the regular representation of GF(2^n).

Let E = (1, x, ..., x^(n-1)) be the standard polynomial basis and let
B = (b_0, ..., b_{n-1}) be a user-selected coordinate basis.

This module follows the construction

    rho_E(a) = sum_i a_i rho_E(x^i),
    rho_B(a) = S^{-1} rho_E(a) S,

where S = [b_0 ... b_{n-1}]_E is the change-of-basis matrix.
"""

from __future__ import annotations

from ..._bmmpy import BitVector as _NativeBitVector, MastrovitoCore as _NativeMastrovitoCore
from .conversions import (
    _int_to_bitvector,
    basis_element_to_int,
    basis_mask_to_field_element,
    build_basis_change_matrix,
    field_element_to_basis_mask,
    field_element_to_int,
    invert_matrix,
)
from .parsing import BasisLike, FieldElementLike, PolynomialLike, parse_field_element, parse_poly


def _nonnegative_int_to_bitvector(value: int) -> _NativeBitVector:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an int")
    if value < 0:
        raise ValueError("value must be non-negative")
    return _int_to_bitvector(value, max(1, value.bit_length()))


class Mastrovito:
    """Build Mastrovito blocks and parity-check matrices for a fixed polynomial.

    Parameters
    ----------
    poly : str | tuple[int, Sequence[int]] | Sequence[int]
        Irreducible polynomial representation.
    basis : Sequence[int | str | Sequence[int]] | None, default=None
        Ordered coordinate basis for the returned matrices.
    element : int | str | Sequence[int] | None, default=None
        Non-zero field element in the standard polynomial basis.
    element_mask : int | None, default=None
        Bitmask of coordinates in the active basis.

    Notes
    -----
    Exactly one of element and element_mask may be passed. If neither is
    provided, the default generator element is x.
    """
    __slots__ = (
        "degree",
        "powers",
        "basis",
        "element",
        "element_mask",
        "element_standard_mask",
        "_period",
        "_impl",
    )

    def __init__(
        self,
        poly: PolynomialLike,
        *,
        basis: BasisLike | None = None,
        element: FieldElementLike | None = None,
        element_mask: int | None = None,
    ) -> None:
        degree, powers = parse_poly(poly)

        if degree <= 0:
            raise ValueError("Polynomial degree must be positive")

        if element is not None and element_mask is not None:
            raise ValueError("Pass either element or element_mask, not both")

        if element is None and element_mask is None:
            element = "x"

        change_inv = None
        basis_bits: list[_NativeBitVector] = []
        if basis is not None:
            change = build_basis_change_matrix(basis, degree, powers)
            change_inv = invert_matrix(change)
            basis_bits = [
                _int_to_bitvector(basis_element_to_int(item, degree, powers), degree)
                for item in basis
            ]

        if element_mask is not None:
            element_bits = basis_mask_to_field_element(
                element_mask,
                basis,
                degree,
                powers,
            )
        else:
            element_bits = field_element_to_int(element, degree, powers)

        if element_bits == 0:
            raise ValueError("element must be non-zero")

        resolved_element_mask = field_element_to_basis_mask(
            element_bits,
            basis,
            degree,
            powers,
            change_inv=change_inv,
        )
        normalized_element = parse_field_element(element_bits)

        self.degree = degree
        self.powers = powers
        self.basis = basis
        self.element = normalized_element
        self.element_mask = resolved_element_mask
        self.element_standard_mask = element_bits
        self._period = (1 << degree) - 1
        self._impl = _NativeMastrovitoCore(
            degree,
            list(powers),
            _int_to_bitvector(element_bits, degree),
            basis_bits,
        )

    def __repr__(self) -> str:
        return (
            f"Mastrovito(degree={self.degree}, powers={self.powers}, "
            f"basis={self.basis!r}, element={self.element!r}, "
            f"element_mask={self.element_mask!r}, "
            f"element_standard_mask={self.element_standard_mask!r})"
        )

    def get_basis_multiplication_matrices(self):
        """Return [rho_B(1), rho_B(x), ..., rho_B(x^(n-1))]."""
        return self._impl.get_basis_multiplication_matrices()

    def get_mastrovito_matrix(self, power: int):
        """Return rho_B(element^power) as a degree x degree matrix."""
        if not isinstance(power, int):
            raise TypeError("power must be an int")
        return self._impl.get_mastrovito_matrix(
            _nonnegative_int_to_bitvector(power % self._period)
        )

    def build_check_matrix(self, c: int, k: int, *, start_i: int = 0):
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
        ...
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

        return self._impl.build_check_matrix(
            c,
            k,
            _nonnegative_int_to_bitvector(start_i),
        )
    
def build_check_matrix(
    poly: PolynomialLike,
    c: int,
    k: int,
    *,
    start_i: int = 0,
    basis: BasisLike | None = None,
    element: FieldElementLike | None = None,
    element_mask: int | None = None,
):
    """Build a Mastrovito-based parity-check matrix for the given field setup."""
    generator = Mastrovito(
        poly,
        basis=basis,
        element=element,
        element_mask=element_mask,
    )
    return generator.build_check_matrix(c, k, start_i=start_i)


def get_mastrovito_matrix(
    poly: PolynomialLike,
    power: int,
    *,
    basis: BasisLike | None = None,
    element: FieldElementLike | None = None,
    element_mask: int | None = None,
):
    """Return the Mastrovito multiplication matrix for the requested power."""
    generator = Mastrovito(
        poly,
        basis=basis,
        element=element,
        element_mask=element_mask,
    )
    return generator.get_mastrovito_matrix(power)


__all__ = [
    "Mastrovito",
    "build_check_matrix",
    "get_mastrovito_matrix",
]
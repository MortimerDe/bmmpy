"""Parsing utilities for polynomials and GF(2^n) field elements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

PolynomialLike: TypeAlias = str | tuple[int, Sequence[int]] | Sequence[int]
FieldElementLike: TypeAlias = str | Sequence[int]
BasisElementLike: TypeAlias = int | FieldElementLike
BasisLike: TypeAlias = Sequence[BasisElementLike]


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


def parse_field_element(element: FieldElementLike) -> list[int]:
    if isinstance(element, str):
        return _parse_field_element_string(element)

    if isinstance(element, Sequence) and not isinstance(element, (bytes, bytearray, str)):
        return _normalize_field_element_powers(element)

    raise TypeError("element must be a polynomial string or sequence of powers")


def _parse_poly_string(poly: str) -> tuple[int, list[int]]:
    powers = _parse_symbolic_powers(poly, label="Polynomial", allow_zero=False)
    return _normalize_powers(powers)


def _parse_field_element_string(element: str) -> list[int]:
    powers = _parse_symbolic_powers(element, label="Field element", allow_zero=True)
    return _normalize_field_element_powers(powers)


def _parse_symbolic_powers(
    text: str,
    *,
    label: str,
    allow_zero: bool,
) -> list[int]:
    compact = text.replace(" ", "")
    if not compact:
        raise ValueError(f"{label} string is empty")

    terms = [term for term in compact.replace("-", "+").split("+") if term]
    if not terms:
        raise ValueError(f"{label} string is empty")

    powers: list[int] = []
    for term in terms:
        if term == "0":
            continue

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
            raise ValueError(f"Unsupported {label.lower()} term: {term!r}")

        if not exponent_text.isdigit():
            raise ValueError(f"Invalid exponent in {label.lower()} term: {term!r}")

        powers.append(int(exponent_text))

    if not powers and not allow_zero:
        raise ValueError(f"{label} must contain at least one non-zero term")

    return powers


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


def _normalize_field_element_powers(values: Sequence[int]) -> list[int]:
    parity: dict[int, bool] = {}

    for value in values:
        if not isinstance(value, int):
            raise TypeError("Field element powers must be integers")
        if value < 0:
            raise ValueError("Field element powers must be non-negative")

        parity[value] = not parity.get(value, False)

    return sorted(
        (power for power, keep in parity.items() if keep),
        reverse=True,
    )


__all__ = [
    "BasisElementLike",
    "BasisLike",
    "FieldElementLike",
    "PolynomialLike",
    "parse_field_element",
    "parse_poly",
]
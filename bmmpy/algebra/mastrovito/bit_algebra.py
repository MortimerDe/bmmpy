"""Bit-packed regular-representation helpers over GF(2)[x] / p(x).

Matrices are represented as ``list[int]`` where each entry packs one row's bits
(LSB = column 0).

If E = (1, x, ..., x^(n-1)) is the standard polynomial basis, then:

- ``build_x_matrix()`` returns rho_E(x),
- ``build_standard_power_matrices()`` returns
  [rho_E(1), rho_E(x), ..., rho_E(x^(n-1))],
- ``combine_power_matrices()`` forms
  rho_E(a) = sum_i a_i rho_E(x^i)
  from the coordinate bits of ``a`` in the standard basis.
"""

from __future__ import annotations

from collections.abc import Sequence


def identity_rows(degree: int) -> list[int]:
    return [1 << index for index in range(degree)]


def build_x_matrix(degree: int, powers: Sequence[int]) -> list[int]:
    """Return rho_E(x) in the standard polynomial basis."""
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


def build_standard_power_matrices(
    degree: int,
    powers: Sequence[int],
) -> list[list[int]]:
    """Return [rho_E(1), rho_E(x), ..., rho_E(x^(n-1))]."""
    mx = build_x_matrix(degree, powers)
    power_matrices = [identity_rows(degree)]

    for _ in range(1, degree):
        power_matrices.append(mul_matrix_rows(power_matrices[-1], mx, degree))

    return power_matrices


def combine_power_matrices(
    power_matrices: Sequence[Sequence[int]],
    coeff_bits: int,
    degree: int,
) -> list[int]:
    """Return sum_i coeff_i * power_matrices[i] over GF(2).

    ``coeff_bits`` is interpreted in the standard polynomial basis:
    coeff_bits = sum_i coeff_i 2^i.
    """
    if coeff_bits < 0:
        raise ValueError("coeff_bits must be non-negative")

    if coeff_bits >> len(power_matrices):
        raise ValueError("coeff_bits requires more basis powers than provided")

    rows = [0] * degree
    bits = coeff_bits

    while bits:
        low_bit = bits & -bits
        bit_index = low_bit.bit_length() - 1
        source = power_matrices[bit_index]

        for row_index, row_bits in enumerate(source):
            rows[row_index] ^= row_bits

        bits ^= low_bit

    return rows


def build_element_matrix(degree: int, powers: Sequence[int], elem: int) -> list[int]:
    """Return rho_E(elem) in the standard polynomial basis."""
    standard_power_matrices = build_standard_power_matrices(degree, powers)
    return combine_power_matrices(standard_power_matrices, elem, degree)


def mul_matrix_rows(left: Sequence[int], right: Sequence[int], degree: int) -> list[int]:
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


def matrix_power_rows(base: Sequence[int], power: int, degree: int) -> list[int]:
    result = identity_rows(degree)
    factor = list(base)
    exponent = power

    while exponent:
        if exponent & 1:
            result = mul_matrix_rows(result, factor, degree)

        exponent >>= 1
        if exponent:
            factor = mul_matrix_rows(factor, factor, degree)

    return result


def precompute_blocks(
    base: Sequence[int],
    degree: int,
    powers: set[int],
) -> dict[int, list[int]]:
    """Cache the requested powers of a single generator matrix."""
    targets = sorted(powers)
    blocks: dict[int, list[int]] = {}

    current_power = 0
    current_rows = identity_rows(degree)

    for target in targets:
        while current_power < target:
            current_rows = mul_matrix_rows(current_rows, base, degree)
            current_power += 1

        blocks[target] = current_rows.copy()

    return blocks


def powers_to_field_element(
    powers: Sequence[int],
    degree: int,
    poly_powers: Sequence[int],
) -> int:
    value = 0
    for power in powers:
        value ^= reduce_monomial(power, degree, poly_powers)
    return value


def reduce_monomial(power: int, degree: int, poly_powers: Sequence[int]) -> int:
    if power < 0:
        raise ValueError("basis powers must be non-negative")

    value = 1 << power
    modulus = 0
    for poly_power in poly_powers:
        modulus |= 1 << poly_power

    while value.bit_length() > degree:
        shift = value.bit_length() - 1 - degree
        value ^= modulus << shift

    return value


__all__ = [
    "build_element_matrix",
    "build_standard_power_matrices",
    "build_x_matrix",
    "combine_power_matrices",
    "identity_rows",
    "matrix_power_rows",
    "mul_matrix_rows",
    "powers_to_field_element",
    "precompute_blocks",
    "reduce_monomial",
]
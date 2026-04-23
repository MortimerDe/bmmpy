"""Bit-packed row arithmetic over GF(2)[x] / p(x).

Matrices are represented as ``list[int]`` where each entry packs one row's bits
(LSB = column 0). All operations are pure-Python and contain no dependency on
:class:`BitMatrix`.
"""

from __future__ import annotations

from collections.abc import Sequence


def identity_rows(degree: int) -> list[int]:
    return [1 << index for index in range(degree)]


def build_x_matrix(degree: int, powers: Sequence[int]) -> list[int]:
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


def build_element_matrix(degree: int, powers: Sequence[int], elem: int) -> list[int]:
    mx = build_x_matrix(degree, powers)
    mx_powers = [identity_rows(degree)]

    for _ in range(1, degree):
        mx_powers.append(mul_matrix_rows(mx_powers[-1], mx, degree))

    rows = [0] * degree
    for bit_index in range(degree):
        if (elem >> bit_index) & 1:
            source = mx_powers[bit_index]
            for row_index, row_bits in enumerate(source):
                rows[row_index] ^= row_bits

    return rows


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
    "build_x_matrix",
    "identity_rows",
    "matrix_power_rows",
    "mul_matrix_rows",
    "powers_to_field_element",
    "precompute_blocks",
    "reduce_monomial",
]
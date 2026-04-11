"""
Matrix types and construction helpers for bmmpy.

BitMatrix is the central data structure in bmmpy. It stores a binary matrix and
supports algebraic operations, serialization, indexing, and row-based updates.

Use matrix_from_rows for small hand-written matrices in examples and tests.
Use BitMatrix constructors and load methods for programmatic or file-based
creation.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._bmmpy import BitMatrix


def matrix_from_rows(rows: Sequence[str]) -> BitMatrix:
    """Build a BitMatrix from equal-width strings of 0 and 1.

    Args:
        rows: Matrix rows encoded as strings such as ["101", "010"].

    Returns:
        A new BitMatrix with one matrix row per input string.

    Raises:
        ValueError: If rows have different widths or contain characters other
            than 0 and 1.

    Example:
        >>> matrix_from_rows(["101", "010"]).shape
        (2, 3)
    """
    return BitMatrix.from_rows(list(rows))


__all__ = ["BitMatrix", "matrix_from_rows"]
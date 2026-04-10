from __future__ import annotations

from collections.abc import Sequence

from ._bmmpy import BitMatrix


def matrix_from_rows(rows: Sequence[str]) -> BitMatrix:
    return BitMatrix.from_rows(list(rows))


__all__ = ["BitMatrix", "matrix_from_rows"]
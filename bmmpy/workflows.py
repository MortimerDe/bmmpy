"""
High-level workflows for bmmpy.

These helpers combine search and apply phases into a single user-facing call.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ._bmmpy import ApplyResult, BitMatrix, Candidate


class SupportsSearch(Protocol):
    def search(self, matrix: BitMatrix, window_rows: Iterable[int]) -> list[Candidate]: ...


class SupportsApply(Protocol):
    def apply(
        self,
        matrix: BitMatrix,
        window_rows: Iterable[int],
        candidates: Iterable[Candidate],
    ) -> ApplyResult: ...


def search_apply(
    matrix: BitMatrix,
    window_rows: Iterable[int],
    *,
    searcher: SupportsSearch,
    selector: SupportsApply,
) -> ApplyResult:
    """
    Run search and apply back-to-back on the same row window.

    Args:
        matrix: Matrix to inspect and update.
        window_rows: Row indices that define the active window.
        searcher: Object with a search method returning Candidate objects.
        selector: Object with an apply method consuming the candidates.

    Returns:
        An ApplyResult describing the performed updates.

    Notes:
        This function mutates matrix in place.
    """
    rows = list(window_rows)
    candidates = searcher.search(matrix, rows)
    return selector.apply(matrix, rows, candidates)


__all__ = ["search_apply"]
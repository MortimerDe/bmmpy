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
    rows = list(window_rows)
    candidates = searcher.search(matrix, rows)
    return selector.apply(matrix, rows, candidates)


__all__ = ["search_apply"]
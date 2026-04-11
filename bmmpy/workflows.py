"""
High-level workflows for bmmpy.

These helpers combine search and apply phases into a single user-facing call.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from ._bmmpy import ApplyResult, Candidate, RowWindow


class SupportsSearch(Protocol):
    def search(self, window: RowWindow) -> list[Candidate]: ...


class SupportsApply(Protocol):
    def apply(
        self,
        window: RowWindow,
        candidates: Iterable[Candidate],
    ) -> ApplyResult: ...


def search_apply(
    window: RowWindow,
    *,
    searcher: SupportsSearch,
    selector: SupportsApply,
) -> ApplyResult:
    """
    Run search and apply back-to-back on the same row window.

    Args:
        window: Row window to inspect and update.
        searcher: Object with a search method returning Candidate objects.
        selector: Object with an apply method consuming the candidates.

    Returns:
        An ApplyResult describing the performed updates.

    Notes:
        This function mutates the underlying matrix in place.
    """
    candidates = searcher.search(window)
    return selector.apply(window, candidates)


__all__ = ["search_apply"]
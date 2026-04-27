"""
High-level workflows for bmmpy.

These helpers combine search and apply steps into a single user-facing call.
They are useful when you want a compact end-to-end workflow without manually
threading candidate lists through your code.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.BitMatrix(2, 5)
>>> for col in (0, 2, 4):
...     matrix[0, col] = True
...     matrix[1, col] = True
>>> window = matrix.row_window([0, 1])
>>> result = bmm.search_apply(
...     window,
...     searcher=bmm.FwhtSearch(max_rows=16, max_candidates=1),
...     applier=bmm.GreedyApplier(min_gain=1),
... )
>>> result.applied_count >= 1
True
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
    applier: SupportsApply,
) -> ApplyResult:
    """Run search and apply back-to-back on the same row window.

    Parameters
    ----------
    window : RowWindow
        Window to inspect and update.
    searcher : SupportsSearch
        Object with a ``search(window)`` method returning Candidate objects.
    applier : SupportsApply
        Object with an ``apply(window, candidates)`` method.

    Returns
    -------
    ApplyResult
        Summary of the accepted applications.

    Notes
    -----
    This function mutates the underlying matrix in place through ``window``.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.BitMatrix(2, 5)
    >>> for col in (0, 2, 4):
    ...     matrix[0, col] = True
    ...     matrix[1, col] = True
    >>> window = matrix.row_window([0, 1])
    >>> result = bmm.search_apply(
    ...     window,
    ...     searcher=bmm.FwhtSearch(max_rows=16, max_candidates=1),
    ...     applier=bmm.GreedyApplier(min_gain=1),
    ... )
    >>> result.applied_count >= 1
    True
    >>> result.weight_improvement > 0
    True
    """
    candidates = searcher.search(window)
    return applier.apply(window, candidates)

__all__ = ["search_apply"]
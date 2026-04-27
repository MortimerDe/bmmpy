"""
Candidate type for bmmpy search results.

A Candidate represents a subset of rows inside a RowWindow together with the
weight of the resulting linear combination. Search algorithms return Candidate objects, and apply strategies such as GreedyApplier consume them.

Examples
--------
>>> import bmmpy as bmm
>>> candidate = bmm.Candidate.from_u64(0b1011, 7)
>>> list(candidate)
[0, 1, 3]
>>> candidate.rows
[0, 1, 3]
>>> candidate.weight
7

Notes
-----
Candidate row indices are local to the searched RowWindow, not necessarily to the original BitMatrix.
"""

from __future__ import annotations

from ._bmmpy import Candidate

Candidate.__doc__ = """
Candidate row combination produced by a search algorithm.

A Candidate stores a subset of window-local row indices together with the
weight of the corresponding linear combination. Lower weights are generally
better.

Examples
--------
>>> import bmmpy as bmm
>>> candidate = bmm.Candidate.from_u64(0b1011, 7)
>>> 0 in candidate, 2 in candidate
(True, False)
>>> list(candidate)
[0, 1, 3]

Notes
-----
The stored row indices are relative to the searched RowWindow.
"""

__all__ = ["Candidate"]
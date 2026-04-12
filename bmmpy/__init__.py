"""
BMMpy: binary matrix minimization tools for Python.
-------

The public package provides a compact Python API for constructing binary
matrices, searching for low-weight row combinations inside row windows, and
applying row transformations in place.

The usual workflow is to build a BitMatrix, select a RowWindow over the rows of
interest, run a search strategy to obtain Candidate objects, and then apply the
best transformations with GreedySelection or a higher-level helper such as
search_apply.

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
...     selector=bmm.GreedySelection(min_gain=1),
... )
>>> result.applied_count >= 1
True
>>> result.weight_improvement > 0
True
"""
from __future__ import annotations

from ._bmmpy import (
    ApplyResult,
    MatrixErr,
    MatrixError,
    RuntimeFeatures,
    add,
    fixed_weight_masks_u32,
    fixed_weight_masks_u64,
    fwht_i16,
    fwht_i32,
    get_runtime_features,
    get_version,
)
from .apply import GreedySelection
from .candidate import Candidate
from .matrix import BitMatrix, RowWindow, matrix_from_rows
from .search import FwhtSearch, MitmFwhtSearch
from .workflows import search_apply

__all__ = [
    "ApplyResult",
    "BitMatrix",
    "Candidate",
    "FwhtSearch",
    "GreedySelection",
    "MatrixErr",
    "MatrixError",
    "MitmFwhtSearch",
    "RowWindow",
    "RuntimeFeatures",
    "add",
    "fixed_weight_masks_u32",
    "fixed_weight_masks_u64",
    "fwht_i16",
    "fwht_i32",
    "get_runtime_features",
    "get_version",
    "matrix_from_rows",
    "search_apply",
]

__version__ = get_version()
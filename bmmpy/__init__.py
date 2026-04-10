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
from .matrix import BitMatrix, matrix_from_rows
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
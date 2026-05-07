"""
Linear row-space transformation helpers over GF(2).
"""

from __future__ import annotations

from .._bmmpy import RowTransformError as _NativeRowTransformError
from .._bmmpy import find_row_transform as _native_find_row_transform
from ..matrix import BitMatrix

RowTransformError = _NativeRowTransformError
RowTransformError.__doc__ = (
    "Raised when a target matrix is not in the row span of a source matrix."
)

def find_row_transform(source: BitMatrix, target: BitMatrix) -> BitMatrix:
    """Return T such that T @ source == target.

    Parameters
    ----------
    source : BitMatrix
        Source matrix A.
    target : BitMatrix
        Target matrix B.

    Returns
    -------
    BitMatrix
        Transformation matrix T with shape (target.rows, source.rows).

    Raises
    ------
    ValueError
        If source and target have different column counts.
    RowTransformError
        If a target row is not representable as an XOR combination of source rows.
    """
    return _native_find_row_transform(source, target)

__all__ = ["RowTransformError", "find_row_transform"]
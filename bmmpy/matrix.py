"""
Matrix types and construction helpers for bmmpy.

BitMatrix is the main matrix container in bmmpy. RowWindow exposes a view over
selected rows of an existing matrix and is the main input type for search and
apply algorithms. Use matrix_from_rows for small hand-written matrices in
examples, tests, and exploratory work.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.matrix_from_rows(["101", "010", "111"])
>>> matrix.shape
(3, 3)
>>> window = matrix.row_window([2, 0])
>>> window.rows
[2, 0]
>>> window.materialize().to_rows()
['111', '101']
"""

from __future__ import annotations

from collections.abc import Sequence

from ._bmmpy import BitMatrix, RowWindow

BitMatrix.__doc__ = """
Packed binary matrix with row-oriented operations over GF(2).

BitMatrix stores boolean values compactly and provides indexing, copying,
serialization, algebraic operations, and extraction of row windows for search
algorithms.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.matrix_from_rows(["10", "01"])
>>> matrix[0, 1] = True
>>> matrix.to_rows()
['11', '01']
>>> matrix.row_window([1, 0]).materialize().to_rows()
['01', '11']

Notes
-----
Matrix arithmetic is performed over GF(2): addition corresponds to XOR, and
matrix multiplication uses binary arithmetic.
"""

RowWindow.__doc__ = """
View over a selected subset of rows in a BitMatrix.

A RowWindow does not own matrix data. It references rows from an existing
matrix and preserves the order of the selected row indices.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.matrix_from_rows(["101", "010", "111"])
>>> window = matrix.row_window([2, 0])
>>> len(window), window.cols, window.rows
(2, 3, [2, 0])
>>> window.materialize().to_rows()
['111', '101']

Notes
-----
Operations that consume a RowWindow may update the underlying matrix in place.
"""

def matrix_from_rows(rows: Sequence[str]) -> BitMatrix:
    """Build a BitMatrix from equal-width strings of 0 and 1.

    Parameters
    ----------
    rows : Sequence[str]
        Matrix rows encoded as strings such as ``["101", "010"]``.

    Returns
    -------
    BitMatrix
        New matrix with one row per input string.

    Raises
    ------
    ValueError
        If the rows have different widths or contain characters other than ``0``
        and ``1``.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.matrix_from_rows(["101", "010"])
    >>> matrix.shape
    (2, 3)
    >>> matrix.to_rows()
    ['101', '010']
    """
    return BitMatrix.from_rows(list(rows))


__all__ = ["BitMatrix", "RowWindow", "matrix_from_rows"]
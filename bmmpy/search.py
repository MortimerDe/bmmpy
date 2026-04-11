"""
Search strategies for bmmpy.

Searchers inspect a RowWindow and return Candidate objects ordered by
nondecreasing weight. The resulting candidates can then be applied to the same window with GreedySelection or another compatible selector.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.BitMatrix(2, 5)
>>> for col in (0, 2, 4):
...     matrix[0, col] = True
...     matrix[1, col] = True
>>> window = matrix.row_window([0, 1])
>>> candidates = bmm.FwhtSearch(max_rows=16, k=1).search(window)
>>> candidates[0].mask_u64(), candidates[0].weight
(3, 0)
"""

from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    BitMatrix,
    Candidate,
    RowWindow,
    FwhtSearch as _NativeFwhtSearch,
    FwhtSearchConfig as _NativeFwhtSearchConfig,
    MitmFwhtSearch as _NativeMitmFwhtSearch,
    MitmFwhtSearchConfig as _NativeMitmFwhtSearchConfig,
)


class FwhtSearch:
    """
    Search for low-weight row combinations using a direct FWHT-based algorithm.

    Parameters
    ----------
    max_rows : int, default=16
        Maximum number of rows allowed in the search window.
    k : int, default=64
        Maximum number of candidates to return.

    Notes
    -----
    This searcher is intended for smaller windows where a direct
    Fast Walsh-Hadamard Transform based approach is appropriate.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.BitMatrix(2, 5)
    >>> for col in (0, 2, 4):
    ...     matrix[0, col] = True
    ...     matrix[1, col] = True
    >>> window = matrix.row_window([0, 1])
    >>> searcher = bmm.FwhtSearch(max_rows=16, k=1)
    >>> candidates = searcher.search(window)
    >>> candidates[0].mask_u64(), candidates[0].weight
    (3, 0)

    See Also
    --------
    MitmFwhtSearch
        Alternative strategy intended for larger search windows.
    """

    __slots__ = ("max_rows", "k", "_impl")

    def __init__(self, *, max_rows: int = 16, k: int = 64) -> None:
        config = _NativeFwhtSearchConfig()
        config.max_rows = max_rows
        config.k = k

        self.max_rows = max_rows
        self.k = k
        self._impl = _NativeFwhtSearch(config)

    def __repr__(self) -> str:
        return f"FwhtSearch(max_rows={self.max_rows}, k={self.k})"

    def name(self) -> str:
        """Return the short native name of the configured search algorithm.

        Returns
        -------
        str
            Algorithm identifier such as ``"fwht"``.
        """
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        """Return a short textual description for a given window size.

        Parameters
        ----------
        window_size : int
            Number of rows in the window that will be searched.

        Returns
        -------
        str
            Human-readable description of the configured strategy.
        """
        return self._impl.describe(window_size)

    def search(self, window: RowWindow) -> list[Candidate]:
        """Search a row window and return candidate row combinations.

        Parameters
        ----------
        window : RowWindow
            Window to search.

        Returns
        -------
        list[Candidate]
            Candidates ordered by nondecreasing weight. Lower weights correspond to
            more attractive row combinations.

        Notes
        -----
        Candidate row indices are local to ``window``.

        Examples
        --------
        >>> import bmmpy as bmm
        >>> matrix = bmm.BitMatrix(2, 5)
        >>> for col in (0, 2, 4):
        ...     matrix[0, col] = True
        ...     matrix[1, col] = True
        >>> window = matrix.row_window([0, 1])
        >>> candidates = bmm.FwhtSearch(max_rows=16, k=1).search(window)
        >>> len(candidates)
        1
        >>> candidates[0].rows
        [0, 1]
        """
        return self._impl.search(window)


class MitmFwhtSearch:
    """
    Search for low-weight row combinations using a meet-in-the-middle modified FWHT algorithm.

    Parameters
    ----------
    initial_capacity_cols : int, default=1024
        Initial internal capacity for packed columns.
    max_t_left : int, default=20
        Initial limit for the left-side split used by the algorithm.
    max_n_right : int, default=65536
        Initial limit for the right-side FWHT search space.
    k : int, default=64
        Maximum number of candidates to return.

    Notes
    -----
    This searcher is intended for larger windows where the direct FWHT approach is
    less suitable.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.matrix_from_rows([
    ...     "110010101001",
    ...     "101100101100",
    ...     "011010011001",
    ...     "111000010111",
    ...     "000111100011",
    ...     "101011110000",
    ... ])
    >>> window = matrix.row_window([0, 1, 2, 3, 4, 5])
    >>> candidates = bmm.MitmFwhtSearch(k=8).search(window)
    >>> len(candidates) > 0
    True

    See Also
    --------
    FwhtSearch
        Direct FWHT strategy for smaller search windows.
    """

    __slots__ = (
        "initial_capacity_cols",
        "max_t_left",
        "max_n_right",
        "k",
        "_impl",
    )

    def __init__(
        self,
        *,
        initial_capacity_cols: int = 1024,
        max_t_left: int = 20,
        max_n_right: int = 1 << 16,
        k: int = 64,
    ) -> None:
        config = _NativeMitmFwhtSearchConfig()
        config.initial_capacity_cols = initial_capacity_cols
        config.max_t_left = max_t_left
        config.max_n_right = max_n_right
        config.k = k

        self.initial_capacity_cols = initial_capacity_cols
        self.max_t_left = max_t_left
        self.max_n_right = max_n_right
        self.k = k
        self._impl = _NativeMitmFwhtSearch(config)

    def __repr__(self) -> str:
        return (
            "MitmFwhtSearch("
            f"initial_capacity_cols={self.initial_capacity_cols}, "
            f"max_t_left={self.max_t_left}, "
            f"max_n_right={self.max_n_right}, "
            f"k={self.k})"
        )

    def name(self) -> str:
        """Return the short native name of the configured search algorithm.

        Returns
        -------
        str
            Algorithm identifier such as ``"mitm_fwht"``.
        """
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        """Return a short textual description for a given window size.

        Parameters
        ----------
        window_size : int
            Number of rows in the window that will be searched.

        Returns
        -------
        str
            Human-readable description of the configured strategy.
        """
        return self._impl.describe(window_size)

    def search(self, window: RowWindow) -> list[Candidate]:
        """Search a row window and return candidate row combinations.

        Parameters
        ----------
        window : RowWindow
            Window to search.

        Returns
        -------
        list[Candidate]
            Candidates ordered by nondecreasing weight. Lower weights correspond to
            more attractive row combinations.

        Notes
        -----
        Candidate row indices are local to ``window``.

        Examples
        --------
        >>> import bmmpy as bmm
        >>> matrix = bmm.matrix_from_rows([
        ...     "110010101001",
        ...     "101100101100",
        ...     "011010011001",
        ...     "111000010111",
        ...     "000111100011",
        ...     "101011110000",
        ... ])
        >>> window = matrix.row_window([0, 1, 2, 3, 4, 5])
        >>> candidates = bmm.MitmFwhtSearch(k=8).search(window)
        >>> len(candidates) > 0
        True
        """
        return self._impl.search(window)


__all__ = ["FwhtSearch", "MitmFwhtSearch"]
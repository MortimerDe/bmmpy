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
>>> candidates = bmm.FwhtSearch(max_rows=16, max_candidates=1).search(window)
>>> candidates[0].mask_u64(), candidates[0].weight
(3, 0)
"""

from __future__ import annotations

from ._bmmpy import (
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
    max_candidates : int, default=64
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
    >>> searcher = bmm.FwhtSearch(max_rows=16, max_candidates=1)
    >>> candidates = searcher.search(window)
    >>> candidates[0].mask_u64(), candidates[0].weight
    (3, 0)

    See Also
    --------
    MitmFwhtSearch
        Alternative strategy intended for larger search windows.
    """

    __slots__ = ("max_rows", "max_candidates", "_impl")

    def __init__(self, *, max_rows: int = 16, max_candidates: int = 64) -> None:
        config = _NativeFwhtSearchConfig()
        config.max_rows = max_rows
        config.max_candidates = max_candidates

        self.max_rows = max_rows
        self.max_candidates = max_candidates
        self._impl = _NativeFwhtSearch(config)

    def __repr__(self) -> str:
        return f"FwhtSearch(max_rows={self.max_rows}, max_candidates={self.max_candidates})"

    def name(self) -> str:
        """Return the short native name of the configured search algorithm.

        Returns
        -------
        str
            Algorithm identifier such as ``"fwht"``.
        """
        return self._impl.name()

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

        Raises
        -----
        ValueError
            If the window contains more than ``max_rows`` rows.
            
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
        >>> candidates = bmm.FwhtSearch(max_rows=16, max_candidates=1).search(window)
        >>> len(candidates)
        1
        >>> candidates[0].rows
        [0, 1]
        """

        window_size = len(window)
        if window_size > self.max_rows:
            raise ValueError(
                f"Window has {window_size} rows, which exceeds the configured "
                f"max_rows={self.max_rows} limit for this searcher."
            )
        return self._impl.search(window)


class MitmFwhtSearch:
    """
    Search for low-weight row combinations using a meet-in-the-middle modified FWHT algorithm.

    Parameters
    ----------
    max_candidates : int, default=64
        Maximum number of candidates to return.

    Notes
    -----
    This searcher is intended for larger windows where the direct FWHT approach is less suitable.

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
    >>> candidates = bmm.MitmFwhtSearch(max_candidates=8).search(window)
    >>> len(candidates) > 0
    True

    See Also
    --------
    FwhtSearch
        Direct FWHT strategy for smaller search windows.
    """

    __slots__ = (
        "max_candidates",
        "_impl",
    )

    def __init__(
        self,
        *,
        max_candidates: int = 64,
    ) -> None:
        config = _NativeMitmFwhtSearchConfig()
        config.max_candidates = max_candidates

        self.max_candidates = max_candidates
        self._impl = _NativeMitmFwhtSearch(config)

    def __repr__(self) -> str:
        return (
            "MitmFwhtSearch("
            f"max_candidates={self.max_candidates})"
        )

    def name(self) -> str:
        """Return the short native name of the configured search algorithm.

        Returns
        -------
        str
            Algorithm identifier such as ``"mitm_fwht"``.
        """
        return self._impl.name()

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
        >>> candidates = bmm.MitmFwhtSearch(max_candidates=8).search(window)
        >>> len(candidates) > 0
        True
        """
        return self._impl.search(window)


__all__ = ["FwhtSearch", "MitmFwhtSearch"]
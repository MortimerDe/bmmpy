"""
Search strategies for bmmpy.

Searchers inspect a window of rows in a BitMatrix and produce Candidate objects ordered by increasing weight.
The returned candidates can then be applied to the matrix with GreedySelection or another selector.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    BitMatrix,
    Candidate,
    FwhtSearch as _NativeFwhtSearch,
    FwhtSearchConfig as _NativeFwhtSearchConfig,
    MitmFwhtSearch as _NativeMitmFwhtSearch,
    MitmFwhtSearchConfig as _NativeMitmFwhtSearchConfig,
)


class FwhtSearch:
    """
    Search for row combinations using the FWHT-based (Fast Walsh-Hadamard Transform) algorithm.

    Args:
        max_rows: Maximum number of rows allowed in the search window.
        k: Maximum number of candidates to return.

    Use this searcher for smaller windows (<23) where the direct FWHT approach is appropriate.
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
        """Return the native algorithm name."""
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        """Return a short textual description for a given window size."""
        return self._impl.describe(window_size)

    def search(self, matrix: BitMatrix, window_rows: Iterable[int]) -> list[Candidate]:
        """Search a row window and return candidates ordered by increasing weight.

        Args:
            matrix: Source matrix.
            window_rows: Row indices that define the search window.

        Returns:
            A list of Candidate objects sorted by nondecreasing weight.
        """
        return self._impl.search(matrix, list(window_rows))


class MitmFwhtSearch:
    """
    Search for row combinations using the meet-in-the-middle modified FWHT algorithm.

    Args:
        initial_capacity_cols: Initial internal capacity for packed columns.
        max_t_left: Initial limit for the left-side split during search.
        max_n_right: Initial limit for the right-side FWHT space.
        k: Maximum number of candidates to return.

    This searcher is intended for larger windows where a meet-in-the-middle
    strategy is more suitable than the direct FWHT approach.
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
        """Return the native algorithm name."""
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        """Return a short textual description for a given window size."""
        return self._impl.describe(window_size)

    def search(self, matrix: BitMatrix, window_rows: Iterable[int]) -> list[Candidate]:
        """
        Search a row window and return candidates ordered by increasing weight.

        Args:
            matrix: Source matrix.
            window_rows: Row indices that define the search window.

        Returns:
            A list of Candidate objects sorted by nondecreasing weight.
        """
        return self._impl.search(matrix, list(window_rows))


__all__ = ["FwhtSearch", "MitmFwhtSearch"]
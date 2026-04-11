"""
Apply strategies for bmmpy.

Selectors consume Candidate objects produced for a RowWindow and update the
underlying BitMatrix in place. GreedySelection is the current built-in strategy
for accepting beneficial row transformations.

Examples
--------
>>> import bmmpy as bmm
>>> matrix = bmm.BitMatrix(2, 5)
>>> for col in (0, 2, 4):
...     matrix[0, col] = True
...     matrix[1, col] = True
>>> window = matrix.row_window([0, 1])
>>> candidates = bmm.FwhtSearch(max_rows=16, max_candidates=1).search(window)
>>> result = bmm.GreedySelection(min_gain=1).apply(window, candidates)
>>> result.applied_count >= 1
True
"""

from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    ApplyResult,
    Candidate,
    RowWindow,
    GreedySelection as _NativeGreedySelection,
)

ApplyResult.__doc__ = """
Summary of a candidate-application step.

Attributes
----------
applied_count : int
    Number of accepted applications.
weight_improvement : int
    Total reduction in row weight across all accepted applications.
"""

class GreedySelection:
    """
    Apply candidates greedily to reduce row weights in a window.

    Parameters
    ----------
    min_gain : int
        Minimum improvement required for an application to be accepted.
    stochastic : bool, default=False
        If ``True``, choose among valid rows and candidates stochastically.
    seed : int, default=0
        Random seed used when ``stochastic`` is enabled.

    Notes
    -----
    This selector mutates the underlying matrix through the provided RowWindow.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.BitMatrix(2, 5)
    >>> for col in (0, 2, 4):
    ...     matrix[0, col] = True
    ...     matrix[1, col] = True
    >>> window = matrix.row_window([0, 1])
    >>> candidates = bmm.FwhtSearch(max_rows=16, max_candidates=1).search(window)
    >>> selector = bmm.GreedySelection(min_gain=1)
    >>> result = selector.apply(window, candidates)
    >>> result.applied_count >= 1
    True
    """

    __slots__ = ("min_gain", "stochastic", "seed", "_impl")

    def __init__(self, min_gain: int, *, stochastic: bool = False, seed: int = 0) -> None:
        self.min_gain = min_gain
        self.stochastic = stochastic
        self.seed = seed
        self._impl = _NativeGreedySelection(min_gain, stochastic=stochastic, seed=seed)

    def __repr__(self) -> str:
        return (
            "GreedySelection("
            f"min_gain={self.min_gain}, "
            f"stochastic={self.stochastic}, "
            f"seed={self.seed})"
        )

    def apply(
        self,
        window: RowWindow,
        candidates: Iterable[Candidate],
    ) -> ApplyResult:
        """Apply candidates to the rows in a window.

        Parameters
        ----------
        window : RowWindow
            Window whose underlying matrix rows may be updated.
        candidates : Iterable[Candidate]
            Candidate objects to evaluate.

        Returns
        -------
        ApplyResult
            Summary of the accepted applications, including the number of applied
            updates and the total weight improvement.

        Notes
        -----
        The underlying matrix is mutated in place.

        Examples
        --------
        >>> import bmmpy as bmm
        >>> matrix = bmm.BitMatrix(2, 5)
        >>> for col in (0, 2, 4):
        ...     matrix[0, col] = True
        ...     matrix[1, col] = True
        >>> window = matrix.row_window([0, 1])
        >>> candidates = bmm.FwhtSearch(max_rows=16, max_candidates=1).search(window)
        >>> result = bmm.GreedySelection(min_gain=1).apply(window, candidates)
        >>> result.weight_improvement > 0
        True
        """
        return self._impl.apply(window, list(candidates))


__all__ = ["GreedySelection"]
"""
Apply strategies for bmmpy.

Selectors consume Candidate objects and update a BitMatrix in place.
GreedySelection is the current built-in strategy for choosing beneficial row
transformations.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    ApplyResult,
    BitMatrix,
    Candidate,
    RowWindow,
    GreedySelection as _NativeGreedySelection,
)


class GreedySelection:
    """
    Apply candidates greedily to reduce row weight.

    Args:
        min_gain: Minimum required improvement for an application to be accepted.
        stochastic: If true, randomize candidate and row choice among valid options.
        seed: Random seed used when stochastic is enabled.

    The selector mutates the input matrix in place.
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
        """
        Apply candidates to a matrix in place.

        Args:
            window: Row window to update through the underlying matrix.
            candidates: Candidate objects to evaluate.
        
        Returns:
            An ApplyResult describing how many changes were applied and  total weight improvement.

        Notes:
            This method mutates the underlying matrix in place.
        """
        return self._impl.apply(window, list(candidates))


__all__ = ["GreedySelection"]
from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    ApplyResult,
    BitMatrix,
    Candidate,
    GreedySelection as _NativeGreedySelection,
)


class GreedySelection:
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
        matrix: BitMatrix,
        window_rows: Iterable[int],
        candidates: Iterable[Candidate],
    ) -> ApplyResult:
        return self._impl.apply(matrix, list(window_rows), list(candidates))


__all__ = ["GreedySelection"]
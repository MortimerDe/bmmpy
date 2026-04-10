from __future__ import annotations

from collections.abc import Iterable

from ._bmmpy import (
    BitMatrix,
    Candidate,
    FwhtSearch as _NativeFwhtSearch,
    FwhtSearchConfig,
    MitmFwhtSearch as _NativeMitmFwhtSearch,
    MitmFwhtSearchConfig,
)


class FwhtSearch:
    __slots__ = ("max_rows", "k", "_impl")

    def __init__(self, *, max_rows: int = 16, k: int = 64) -> None:
        config = FwhtSearchConfig()
        config.max_rows = max_rows
        config.k = k

        self.max_rows = max_rows
        self.k = k
        self._impl = _NativeFwhtSearch(config)

    def __repr__(self) -> str:
        return f"FwhtSearch(max_rows={self.max_rows}, k={self.k})"

    def name(self) -> str:
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        return self._impl.describe(window_size)

    def search(self, matrix: BitMatrix, window_rows: Iterable[int]) -> list[Candidate]:
        return self._impl.search(matrix, list(window_rows))


class MitmFwhtSearch:
    __slots__ = (
        "initial_capacity_cols",
        "max_t_left",
        "max_n_right",
        "k_limit",
        "_impl",
    )

    def __init__(
        self,
        *,
        initial_capacity_cols: int = 1024,
        max_t_left: int = 20,
        max_n_right: int = 1 << 16,
        k_limit: int = 64,
    ) -> None:
        config = MitmFwhtSearchConfig()
        config.initial_capacity_cols = initial_capacity_cols
        config.max_t_left = max_t_left
        config.max_n_right = max_n_right
        config.k_limit = k_limit

        self.initial_capacity_cols = initial_capacity_cols
        self.max_t_left = max_t_left
        self.max_n_right = max_n_right
        self.k_limit = k_limit
        self._impl = _NativeMitmFwhtSearch(config)

    def __repr__(self) -> str:
        return (
            "MitmFwhtSearch("
            f"initial_capacity_cols={self.initial_capacity_cols}, "
            f"max_t_left={self.max_t_left}, "
            f"max_n_right={self.max_n_right}, "
            f"k_limit={self.k_limit})"
        )

    def name(self) -> str:
        return self._impl.name()

    def describe(self, window_size: int) -> str:
        return self._impl.describe(window_size)

    def search(self, matrix: BitMatrix, window_rows: Iterable[int]) -> list[Candidate]:
        return self._impl.search(matrix, list(window_rows))


__all__ = ["FwhtSearch", "MitmFwhtSearch"]
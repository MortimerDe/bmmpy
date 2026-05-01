# bmmpy/exact.py
"""
Exact basis optimization helpers for bmmpy.

This module exposes an exact solver for the following objective:

- among all bases of the row space of a window, find one with minimum total row weight

The current implementation is intentionally capped and is meant for small exact
runs such as 24-row windows.
"""

from __future__ import annotations

from ._bmmpy import (
    ExactBasisResult,
    ExactBasisSolver as _NativeExactBasisSolver,
    ExactBasisSolverConfig as _NativeExactBasisSolverConfig,
    RowWindow,
)


ExactBasisSolverConfig = _NativeExactBasisSolverConfig


class ExactBasisSolver:
    __slots__ = ("max_rows", "max_states", "max_storage_bytes", "_impl")

    def __init__(
        self,
        *,
        max_rows: int = 24,
        max_states: int = (1 << 24) - 1,
        max_storage_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        config = _NativeExactBasisSolverConfig()
        config.max_rows = max_rows
        config.max_states = max_states
        config.max_storage_bytes = max_storage_bytes

        self.max_rows = max_rows
        self.max_states = max_states
        self.max_storage_bytes = max_storage_bytes
        self._impl = _NativeExactBasisSolver(config)

    def __repr__(self) -> str:
        return (
            "ExactBasisSolver("
            f"max_rows={self.max_rows}, "
            f"max_states={self.max_states}, "
            f"max_storage_bytes={self.max_storage_bytes})"
        )

    def name(self) -> str:
        return self._impl.name()

    def solve(self, window: RowWindow) -> ExactBasisResult:
        return self._impl.solve(window)


__all__ = [
    "ExactBasisResult",
    "ExactBasisSolver",
    "ExactBasisSolverConfig",
]
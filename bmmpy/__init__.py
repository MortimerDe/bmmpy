from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._bmmpy import (
    ApplyResult,
    BitMatrix,
    Candidate,
    FwhtSearch,
    FwhtSearchConfig,
    GreedySelection,
    MatrixErr,
    MatrixError,
    MitmFwhtSearch,
    MitmFwhtSearchConfig,
    RuntimeFeatures,
    add,
    fixed_weight_masks_u32,
    fixed_weight_masks_u64,
    fwht_i16,
    fwht_i32,
    get_runtime_features,
    get_version,
)


def matrix_from_rows(rows: Sequence[str]) -> BitMatrix:
    return BitMatrix.from_rows(list(rows))


def make_fwht_config(*, max_rows: int = 16, k: int = 64) -> FwhtSearchConfig:
    config = FwhtSearchConfig()
    config.max_rows = max_rows
    config.k = k
    return config


def make_mitm_fwht_config(
    *,
    initial_capacity_cols: int = 1024,
    max_t_left: int = 20,
    max_n_right: int = 1 << 16,
    k_limit: int = 64,
) -> MitmFwhtSearchConfig:
    config = MitmFwhtSearchConfig()
    config.initial_capacity_cols = initial_capacity_cols
    config.max_t_left = max_t_left
    config.max_n_right = max_n_right
    config.k_limit = k_limit
    return config


def fwht_search(
    matrix: BitMatrix,
    window_rows: Iterable[int],
    *,
    max_rows: int = 16,
    k: int = 64,
) -> list[Candidate]:
    searcher = FwhtSearch(make_fwht_config(max_rows=max_rows, k=k))
    return searcher.search(matrix, list(window_rows))


def mitm_fwht_search(
    matrix: BitMatrix,
    window_rows: Iterable[int],
    *,
    initial_capacity_cols: int = 1024,
    max_t_left: int = 20,
    max_n_right: int = 1 << 16,
    k_limit: int = 64,
) -> list[Candidate]:
    searcher = MitmFwhtSearch(
        make_mitm_fwht_config(
            initial_capacity_cols=initial_capacity_cols,
            max_t_left=max_t_left,
            max_n_right=max_n_right,
            k_limit=k_limit,
        )
    )
    return searcher.search(matrix, list(window_rows))


def apply_greedy(
    matrix: BitMatrix,
    window_rows: Iterable[int],
    candidates: Iterable[Candidate],
    *,
    min_gain: int,
    stochastic: bool = False,
    seed: int = 0,
) -> ApplyResult:
    selector = GreedySelection(min_gain, stochastic=stochastic, seed=seed)
    return selector.apply(matrix, list(window_rows), list(candidates))


__all__ = [
    "ApplyResult",
    "BitMatrix",
    "Candidate",
    "FwhtSearch",
    "FwhtSearchConfig",
    "GreedySelection",
    "MatrixErr",
    "MatrixError",
    "MitmFwhtSearch",
    "MitmFwhtSearchConfig",
    "add",
    "apply_greedy",
    "fixed_weight_masks_u32",
    "fixed_weight_masks_u64",
    "fwht_i16",
    "fwht_i32",
    "fwht_search",
    "get_version",
    "make_fwht_config",
    "make_mitm_fwht_config",
    "matrix_from_rows",
    "mitm_fwht_search",
    "RuntimeFeatures",
    "get_runtime_features",
]

__version__ = get_version()
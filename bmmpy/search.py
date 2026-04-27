"""
Search strategies for bmmpy.

Searchers inspect a RowWindow and return Candidate objects ordered by
nondecreasing weight. The resulting candidates can then be applied to the same window with GreedyApplier or another compatible applier.

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
    BruteforceSearch as _NativeBruteforceSearch,
    BruteforceSearchConfig as _NativeBruteforceSearchConfig,
    CudaBruteforceSearch as _NativeCudaBruteforceSearch,
    CudaBruteforceSearchConfig as _NativeCudaBruteforceSearchConfig,
    CudaMitmFwhtSearch as _NativeCudaMitmFwhtSearch,
    CudaMitmFwhtSearchConfig as _NativeCudaMitmFwhtSearchConfig,
    FwhtSearch as _NativeFwhtSearch,
    FwhtSearchConfig as _NativeFwhtSearchConfig,
    MitmFwhtSearch as _NativeMitmFwhtSearch,
    MitmFwhtSearchConfig as _NativeMitmFwhtSearchConfig,
    SASelectionResult,
    SASelector as _NativeSASelector,
    SASelectorConfig as _NativeSASelectorConfig,
    CoolingPolicyKind as _NativeCoolingPolicyKind,
    WindowScorePolicyKind as _NativeWindowScorePolicyKind,
)

def _resolve_score_policy(name: str):
    if name == "pairwise_synergy":
        return _NativeWindowScorePolicyKind.PairwiseSynergy
    raise ValueError(f"Unsupported score_policy: {name!r}")


def _resolve_cooling_policy(name: str):
    if name == "adaptive_geometric":
        return _NativeCoolingPolicyKind.AdaptiveGeometric
    raise ValueError(f"Unsupported cooling_policy: {name!r}")

class BruteforceSearch:
    """
    Search for low-weight row combinations using exact direct brute force.

    Parameters
    ----------
    max_candidates : int, default=64
        Maximum number of candidates to return.
    chunk_bits : int, default=0
        Width of the Gray-swept low chunk. A value of 0 selects the native auto mode.

    Notes
    -----
    This searcher targets narrow matrices where direct exact search over row
    combinations is competitive. The native implementation partitions the mask
    space by high-prefix, sweeps each chunk in Gray order, and keeps per-thread
    exact top-k results before a final merge.

    See Also
    --------
    FwhtSearch
        Direct FWHT strategy for smaller windows.
    MitmFwhtSearch
        Split-FWHT strategy for larger windows.
    """

    __slots__ = ("max_candidates", "chunk_bits", "_impl")

    def __init__(self, *, max_candidates: int = 64, chunk_bits: int = 0) -> None:
        config = _NativeBruteforceSearchConfig()
        config.max_candidates = max_candidates
        config.chunk_bits = chunk_bits

        self.max_candidates = max_candidates
        self.chunk_bits = chunk_bits
        self._impl = _NativeBruteforceSearch(config)

    def __repr__(self) -> str:
        return (
            "BruteforceSearch("
            f"max_candidates={self.max_candidates}, "
            f"chunk_bits={self.chunk_bits})"
        )

    def name(self) -> str:
        return self._impl.name()

    def search(self, window: RowWindow) -> list[Candidate]:
        return self._impl.search(window)

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

class CudaMitmFwhtSearch:
    """
    Search for low-weight row combinations using the CUDA split-FWHT backend.

    Parameters
    ----------
    max_candidates : int, default=64
        Maximum number of candidates to return.
    low_bits : int, default=0
        Size of the low-half split. A value of 0 means auto.

    Notes
    -----
    This searcher is intended for GPU execution on larger row windows.
    The current native implementation validates GPU-compatible windows and
    dispatches to the CUDA backend when available.

    See Also
    --------
    MitmFwhtSearch
        CPU split searcher for larger windows.
    FwhtSearch
        Direct FWHT strategy for smaller windows.
    """

    __slots__ = (
        "max_candidates",
        "low_bits",
        "_impl",
    )

    def __init__(self, *, max_candidates: int = 64, low_bits: int = 0) -> None:
        config = _NativeCudaMitmFwhtSearchConfig()
        config.max_candidates = max_candidates
        config.low_bits = low_bits

        self.max_candidates = max_candidates
        self.low_bits = low_bits
        self._impl = _NativeCudaMitmFwhtSearch(config)

    def __repr__(self) -> str:
        return (
            "CudaMitmFwhtSearch("
            f"max_candidates={self.max_candidates}, "
            f"low_bits={self.low_bits})"
        )

    def name(self) -> str:
        return self._impl.name()

    def search(self, window: RowWindow) -> list[Candidate]:
        return self._impl.search(window)

class CudaBruteforceSearch:
    """
    Search for low-weight row combinations using the CUDA exact brute-force backend.

    Parameters
    ----------
    max_candidates : int, default=64
        Maximum number of candidates to return.
    chunk_bits : int, default=0
        Width of the Gray-swept low chunk. A value of 0 selects the native auto mode.

    Notes
    -----
    This searcher targets GPU execution for narrow matrices in the current CUDA
    brute-force path. The native implementation validates GPU-compatible windows
    and dispatches to the CUDA backend when available.

    See Also
    --------
    BruteforceSearch
        CPU exact brute-force searcher for narrow matrices.
    CudaMitmFwhtSearch
        CUDA split-FWHT backend for larger-window GPU search.
    """

    __slots__ = ("max_candidates", "chunk_bits", "_impl")

    def __init__(self, *, max_candidates: int = 64, chunk_bits: int = 0) -> None:
        config = _NativeCudaBruteforceSearchConfig()
        config.max_candidates = max_candidates
        config.chunk_bits = chunk_bits

        self.max_candidates = max_candidates
        self.chunk_bits = chunk_bits
        self._impl = _NativeCudaBruteforceSearch(config)

    def __repr__(self) -> str:
        return (
            "CudaBruteforceSearch("
            f"max_candidates={self.max_candidates}, "
            f"chunk_bits={self.chunk_bits})"
        )

    def name(self) -> str:
        return self._impl.name()

    def search(self, window: RowWindow) -> list[Candidate]:
        return self._impl.search(window)

class SASelector:
    """
    Select promising row windows using simulated annealing on the CPU.

    Parameters
    ----------
    iterations : int, default=10000
        Number of annealing steps per restart.
    restarts : int, default=8
        Number of independent restarts. The best visited state is returned.
    seed : int, default=0
        Seed for the native pseudo-random generator.
    score_policy : str, default="pairwise_synergy"
        Window scoring policy. The current implementation supports only
        ``"pairwise_synergy"``.
    cooling_policy : str, default="adaptive_geometric"
        Cooling schedule policy. The current implementation supports only
        ``"adaptive_geometric"``.
    temperature_probe_samples : int, default=64
        Number of sampled negative deltas used to estimate the initial temperature.
    initial_acceptance_probability : float, default=0.8
        Target initial acceptance probability for sampled negative moves.
    cooling_rate : float, default=0.99
        Geometric cooling multiplier applied after each iteration.
    min_temperature : float, default=1e-6
        Lower clamp for the temperature.

    Notes
    -----
    This selector is a preprocessing step. It does not run the search itself.
    Instead, it chooses a row window that is likely to be useful for a downstream
    search backend.

    Examples
    --------
    >>> import bmmpy as bmm
    >>> matrix = bmm.matrix_from_rows([
    ...     "111111000000",
    ...     "111111000000",
    ...     "111111000000",
    ...     "111111000000",
    ...     "000000000000",
    ...     "000000000000",
    ... ])
    >>> selector = bmm.SASelector(iterations=256, restarts=4, seed=7)
    >>> result = selector.select(matrix, 4)
    >>> len(result.rows)
    4
    >>> window = selector.select_window(matrix, 4)
    >>> len(window)
    4
    """

    __slots__ = (
        "iterations",
        "restarts",
        "seed",
        "score_policy",
        "cooling_policy",
        "temperature_probe_samples",
        "initial_acceptance_probability",
        "cooling_rate",
        "min_temperature",
        "_impl",
    )

    def __init__(
        self,
        *,
        iterations: int = 10000,
        restarts: int = 8,
        seed: int = 0,
        score_policy: str = "pairwise_synergy",
        cooling_policy: str = "adaptive_geometric",
        temperature_probe_samples: int = 64,
        initial_acceptance_probability: float = 0.8,
        cooling_rate: float = 0.99,
        min_temperature: float = 1e-6,
    ) -> None:
        config = _NativeSASelectorConfig()
        config.iterations = iterations
        config.restarts = restarts
        config.seed = seed
        config.score_policy = _resolve_score_policy(score_policy)
        config.cooling_policy = _resolve_cooling_policy(cooling_policy)
        config.temperature_probe_samples = temperature_probe_samples
        config.initial_acceptance_probability = initial_acceptance_probability
        config.cooling_rate = cooling_rate
        config.min_temperature = min_temperature

        self.iterations = iterations
        self.restarts = restarts
        self.seed = seed
        self.score_policy = score_policy
        self.cooling_policy = cooling_policy
        self.temperature_probe_samples = temperature_probe_samples
        self.initial_acceptance_probability = initial_acceptance_probability
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self._impl = _NativeSASelector(config)

    def __repr__(self) -> str:
        return (
            "SASelector("
            f"iterations={self.iterations}, "
            f"restarts={self.restarts}, "
            f"seed={self.seed}, "
            f"score_policy={self.score_policy!r}, "
            f"cooling_policy={self.cooling_policy!r}, "
            f"temperature_probe_samples={self.temperature_probe_samples}, "
            f"initial_acceptance_probability={self.initial_acceptance_probability}, "
            f"cooling_rate={self.cooling_rate}, "
            f"min_temperature={self.min_temperature})"
        )

    def name(self) -> str:
        return self._impl.name()

    def select(self, matrix, window_size: int) -> SASelectionResult:
        return self._impl.select(matrix, window_size)

    def select_window(self, matrix, window_size: int) -> RowWindow:
        return self._impl.select_window(matrix, window_size)

__all__ = ["BruteforceSearch", "FwhtSearch", "MitmFwhtSearch", "CudaMitmFwhtSearch", "CudaBruteforceSearch", "SASelector", "SASelectionResult"]
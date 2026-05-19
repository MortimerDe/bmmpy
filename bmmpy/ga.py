"""
Genetic-algorithm optimizers for bmmpy.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._bmmpy import (
    Candidate,
    GeneticAlgorithm as _NativeGeneticAlgorithm,
    GeneticAlgorithmConfig,
    IslandModel as _NativeIslandModel,
    IslandModelConfig,
    IslandModelSnapshot,
    IslandSnapshot,
    IslandSpec,
    MigrationPolicy,
    OverflowPolicy as _NativeOverflowPolicy,
    RowWindow,
    RunStats,
    StopCriteria,
)

OverflowPolicy = _NativeOverflowPolicy

StopCriteria.__doc__ = "Stopping rules for the genetic optimizers."
RunStats.__doc__ = "Runtime counters collected during a genetic run."
MigrationPolicy.__doc__ = "Per-island migration settings."
IslandSpec.__doc__ = "Static configuration for one island in the island model."
IslandSnapshot.__doc__ = "Per-island runtime snapshot."
IslandModelSnapshot.__doc__ = "Whole-model runtime snapshot."
GeneticAlgorithmConfig.__doc__ = "Low-level native GA configuration."
IslandModelConfig.__doc__ = "Low-level native island-model configuration."


def _build_stop_criteria(
    *,
    max_generations: int | None,
    max_stale_generations: int | None,
    target_total_weight: int | None,
) -> StopCriteria:
    stop = StopCriteria()
    stop.max_generations = max_generations
    stop.max_stale_generations = max_stale_generations
    stop.target_total_weight = target_total_weight
    return stop


def _clone_stop_criteria(stop: StopCriteria) -> StopCriteria:
    clone = StopCriteria()
    clone.max_generations = stop.max_generations
    clone.max_stale_generations = stop.max_stale_generations
    clone.target_total_weight = stop.target_total_weight
    return clone


def _build_ga_config(
    *,
    population_size: int,
    elite_count: int,
    tournament_size: int,
    mutation_rate: float,
    max_generations: int | None,
    max_stale_generations: int | None,
    target_total_weight: int | None,
    seed: int,
) -> GeneticAlgorithmConfig:
    config = GeneticAlgorithmConfig()
    config.population_size = population_size
    config.elite_count = elite_count
    config.tournament_size = tournament_size
    config.mutation_rate = mutation_rate
    config.stop = _build_stop_criteria(
        max_generations=max_generations,
        max_stale_generations=max_stale_generations,
        target_total_weight=target_total_weight,
    )
    config.seed = seed
    return config


def _clone_ga_config(config: GeneticAlgorithmConfig) -> GeneticAlgorithmConfig:
    clone = GeneticAlgorithmConfig()
    clone.population_size = config.population_size
    clone.elite_count = config.elite_count
    clone.tournament_size = config.tournament_size
    clone.mutation_rate = config.mutation_rate
    clone.stop = _clone_stop_criteria(config.stop)
    clone.seed = config.seed
    return clone


def _resolve_overflow_policy(
    policy: str | OverflowPolicy,
) -> OverflowPolicy:
    if isinstance(policy, str):
        normalized = policy.strip().lower()
        if normalized == "drop_oldest":
            return OverflowPolicy.DropOldest
        if normalized == "drop_newest":
            return OverflowPolicy.DropNewest
        if normalized == "reject_publish":
            return OverflowPolicy.RejectPublish
        raise ValueError(f"Unsupported overflow_policy: {policy!r}")

    if policy not in (
        OverflowPolicy.DropOldest,
        OverflowPolicy.DropNewest,
        OverflowPolicy.RejectPublish,
    ):
        raise ValueError(f"Unsupported overflow_policy: {policy!r}")

    return policy


def _common_ga_kwargs_are_default(
    *,
    population_size: int,
    elite_count: int,
    tournament_size: int,
    mutation_rate: float,
    max_generations: int | None,
    max_stale_generations: int | None,
    target_total_weight: int | None,
    seed: int,
    seed_stride: int,
) -> bool:
    return (
        population_size == 300
        and elite_count == 3
        and tournament_size == 3
        and mutation_rate == 0.3
        and max_generations is None
        and max_stale_generations is None
        and target_total_weight is None
        and seed == 0
        and seed_stride == 1
    )


class GeneticAlgorithm:
    """
    Genetic optimizer over a row window.

    The returned value is a full list of Candidate objects with length equal to
    the input window size. Each candidate describes one output row of the found
    transform.
    """

    __slots__ = (
        "population_size",
        "elite_count",
        "tournament_size",
        "mutation_rate",
        "max_generations",
        "max_stale_generations",
        "target_total_weight",
        "seed",
        "_impl",
    )

    def __init__(
        self,
        *,
        population_size: int = 300,
        elite_count: int = 3,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        max_generations: int | None = None,
        max_stale_generations: int | None = None,
        target_total_weight: int | None = None,
        seed: int = 0,
    ) -> None:
        self.population_size = population_size
        self.elite_count = elite_count
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_stale_generations = max_stale_generations
        self.target_total_weight = target_total_weight
        self.seed = seed

        config = _build_ga_config(
            population_size=population_size,
            elite_count=elite_count,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            max_generations=max_generations,
            max_stale_generations=max_stale_generations,
            target_total_weight=target_total_weight,
            seed=seed,
        )
        self._impl = _NativeGeneticAlgorithm(config)

    def __repr__(self) -> str:
        return (
            "GeneticAlgorithm("
            f"population_size={self.population_size}, "
            f"elite_count={self.elite_count}, "
            f"tournament_size={self.tournament_size}, "
            f"mutation_rate={self.mutation_rate}, "
            f"max_generations={self.max_generations}, "
            f"max_stale_generations={self.max_stale_generations}, "
            f"target_total_weight={self.target_total_weight}, "
            f"seed={self.seed})"
        )

    def name(self) -> str:
        return self._impl.name()

    def initialize(self, window: RowWindow) -> None:
        self._impl.initialize(window)

    def step(self) -> None:
        self._impl.step()

    def done(self) -> bool:
        return self._impl.done()

    def generation(self) -> int:
        return self._impl.generation()

    def best_individual(self) -> list[Candidate]:
        return self._impl.best_individual()

    def stats(self) -> RunStats:
        return self._impl.stats()

    def best_score(self) -> int:
        return self._impl.best_score()

    def optimize(self, window: RowWindow) -> list[Candidate]:
        return self._impl.optimize(window)


class IslandModel:
    """
    Island-model wrapper around GeneticAlgorithm.

    By default this class builds one GA configuration per island from the common
    keyword arguments. For heterogeneous islands you can pass ga_configs with
    one GeneticAlgorithmConfig per IslandSpec.
    """

    __slots__ = (
        "island_count",
        "channel_capacity",
        "overflow_policy",
        "_impl",
    )

    def __init__(
        self,
        *,
        islands: Sequence[IslandSpec],
        ga_configs: Sequence[GeneticAlgorithmConfig] | None = None,
        population_size: int = 300,
        elite_count: int = 3,
        tournament_size: int = 3,
        mutation_rate: float = 0.3,
        max_generations: int | None = None,
        max_stale_generations: int | None = None,
        target_total_weight: int | None = None,
        seed: int = 0,
        seed_stride: int = 1,
        channel_capacity: int = 256,
        overflow_policy: str | OverflowPolicy = "drop_oldest",
    ) -> None:
        island_list = list(islands)
        if not island_list:
            raise ValueError("IslandModel requires at least one island")
        if channel_capacity <= 0:
            raise ValueError("channel_capacity must be greater than zero")
        if seed_stride < 0:
            raise ValueError("seed_stride must be non-negative")

        if ga_configs is not None:
            if len(ga_configs) != len(island_list):
                raise ValueError(
                    "ga_configs length must match the number of islands"
                )
            if not _common_ga_kwargs_are_default(
                population_size=population_size,
                elite_count=elite_count,
                tournament_size=tournament_size,
                mutation_rate=mutation_rate,
                max_generations=max_generations,
                max_stale_generations=max_stale_generations,
                target_total_weight=target_total_weight,
                seed=seed,
                seed_stride=seed_stride,
            ):
                raise ValueError(
                    "Pass either ga_configs or common GA keyword arguments, not both"
                )
            normalized_ga_configs = [_clone_ga_config(config) for config in ga_configs]
        else:
            normalized_ga_configs = [
                _build_ga_config(
                    population_size=population_size,
                    elite_count=elite_count,
                    tournament_size=tournament_size,
                    mutation_rate=mutation_rate,
                    max_generations=max_generations,
                    max_stale_generations=max_stale_generations,
                    target_total_weight=target_total_weight,
                    seed=seed + index * seed_stride,
                )
                for index in range(len(island_list))
            ]

        config = IslandModelConfig()
        config.islands = island_list

        resolved_overflow_policy = _resolve_overflow_policy(overflow_policy)

        self.island_count = len(island_list)
        self.channel_capacity = channel_capacity
        self.overflow_policy = resolved_overflow_policy
        self._impl = _NativeIslandModel(
            config,
            normalized_ga_configs,
            channel_capacity,
            resolved_overflow_policy,
        )

    def __repr__(self) -> str:
        return (
            "IslandModel("
            f"island_count={self.island_count}, "
            f"channel_capacity={self.channel_capacity}, "
            f"overflow_policy={self.overflow_policy})"
        )

    def name(self) -> str:
        return self._impl.name()

    def initialize(self, window: RowWindow) -> None:
        self._impl.initialize(window)

    def start(self) -> None:
        self._impl.start()

    def request_stop(self) -> None:
        self._impl.request_stop()

    def wait(self) -> None:
        self._impl.wait()

    def running(self) -> bool:
        return self._impl.running()

    def stop_requested(self) -> bool:
        return self._impl.stop_requested()

    def best_individual(self) -> list[Candidate]:
        return self._impl.best_individual()

    def best_score(self) -> int:
        return self.snapshot().best_score

    def snapshot(self) -> IslandModelSnapshot:
        return self._impl.snapshot()

    def run_to_completion(self, window: RowWindow) -> list[Candidate]:
        return self._impl.run_to_completion(window)


__all__ = [
    "GeneticAlgorithm",
    "GeneticAlgorithmConfig",
    "IslandModel",
    "IslandModelConfig",
    "IslandModelSnapshot",
    "IslandSnapshot",
    "IslandSpec",
    "MigrationPolicy",
    "OverflowPolicy",
    "RunStats",
    "StopCriteria",
]
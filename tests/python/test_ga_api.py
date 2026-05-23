from __future__ import annotations

import unittest

import bmmpy as bmm


def make_ga_matrix() -> bmm.BitMatrix:
    return bmm.matrix_from_rows(
        [
            "11001000",
            "01100100",
            "00100110",
            "10010001",
        ]
    )


def make_island_spec(island_id: int, interval_generations: int) -> bmm.IslandSpec:
    migration = bmm.MigrationPolicy()
    migration.interval_generations = interval_generations
    migration.export_count = 1
    migration.import_count = 1

    spec = bmm.IslandSpec()
    spec.island_id = island_id
    spec.migration = migration
    return spec


def make_native_ga_config(seed: int) -> bmm.ga.GeneticAlgorithmConfig:
    config = bmm.ga.GeneticAlgorithmConfig()
    config.population_size = 12
    config.elite_count = 2
    config.tournament_size = 2
    config.mutation_rate = 0.25
    config.num_parents = 2
    config.num_offspring = 1
    config.enable_catastrophe = True
    config.catastrophe_threshold = 2
    config.catastrophe_survival_rate = 0.5

    stop = bmm.StopCriteria()
    stop.max_generations = 4
    stop.max_stale_generations = 4
    config.stop = stop
    config.seed = seed
    return config


class TestGaApi(unittest.TestCase):
    def test_genetic_algorithm_stepwise_wrapper(self) -> None:
        matrix = make_ga_matrix()

        ga = bmm.GeneticAlgorithm(
            population_size=16,
            elite_count=2,
            tournament_size=2,
            mutation_rate=0.3,
            max_generations=4,
            max_stale_generations=4,
            seed=7,
            num_parents=2,
            num_offspring=1,
            enable_catastrophe=True,
            catastrophe_threshold=2,
            catastrophe_survival_rate=0.5,
        )

        ga.initialize(matrix.row_window([0, 1, 2, 3]))

        while not ga.done():
            ga.step()

        individual = ga.best_individual()
        stats = ga.stats()

        self.assertEqual(ga.name(), "ga")
        self.assertEqual(len(individual), 4)
        self.assertEqual(ga.best_score(), sum(candidate.weight for candidate in individual))
        self.assertEqual(stats.generations, ga.generation())
        self.assertGreater(stats.evaluations, 0)
        self.assertEqual(stats.seed, 7)
        self.assertEqual(ga.num_parents, 2)
        self.assertEqual(ga.num_offspring, 1)
        self.assertTrue(ga.enable_catastrophe)
        self.assertEqual(ga.catastrophe_threshold, 2)
        self.assertEqual(ga.catastrophe_survival_rate, 0.5)

    def test_island_model_wrapper(self) -> None:
        matrix = make_ga_matrix()
        window = matrix.row_window([0, 1, 2, 3])

        islands = [
            make_island_spec(0, 4),
            make_island_spec(1, 6),
        ]
        ga_configs = [
            make_native_ga_config(11),
            make_native_ga_config(19),
        ]

        model = bmm.IslandModel(
            islands=islands,
            ga_configs=ga_configs,
            channel_capacity=32,
            overflow_policy="drop_oldest",
        )

        best = model.run_to_completion(window)
        snapshot = model.snapshot()

        self.assertEqual(model.name(), "island_model")
        self.assertEqual(len(best), 4)
        self.assertEqual(snapshot.island_count, 2)
        self.assertEqual(len(snapshot.islands), 2)
        self.assertEqual(snapshot.best_score, sum(candidate.weight for candidate in best))
        self.assertFalse(snapshot.running)
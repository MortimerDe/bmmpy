from __future__ import annotations

import unittest

import bmmpy as bmm


def make_search_matrix() -> bmm.BitMatrix:
    matrix = bmm.BitMatrix(2, 5)
    for col in (0, 2, 4):
        matrix[0, col] = True
        matrix[1, col] = True
    return matrix


class TestSearchApi(unittest.TestCase):
    def test_fwht_search_wrapper(self) -> None:
        matrix = make_search_matrix()
        window = matrix.row_window([0, 1])
        searcher = bmm.FwhtSearch(max_rows=16, max_candidates=1)

        candidates = searcher.search(window)

        self.assertEqual(searcher.name(), "fwht")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].mask_u64(), 0x3)
        self.assertEqual(candidates[0].weight, 0)

    def test_mitm_search_wrapper(self) -> None:
        matrix = bmm.matrix_from_rows(
            [
                "110010101001",
                "101100101100",
                "011010011001",
                "111000010111",
                "000111100011",
                "101011110000",
            ]
        )

        fwht = bmm.FwhtSearch(max_rows=16, max_candidates=8)
        mitm = bmm.MitmFwhtSearch(max_candidates=8)

        window = matrix.row_window([0, 1, 2, 3, 4, 5])
        expected = fwht.search(window)
        actual = mitm.search(window)

        self.assertEqual(
            [(candidate.mask_u64(), candidate.weight) for candidate in actual],
            [(candidate.mask_u64(), candidate.weight) for candidate in expected],
        )

    def test_bruteforce_search_wrapper(self) -> None:
        matrix = bmm.matrix_from_rows(
            [
                "110010101001",
                "101100101100",
                "011010011001",
                "111000010111",
                "000111100011",
                "101011110000",
            ]
        )

        fwht = bmm.FwhtSearch(max_rows=16, max_candidates=8)
        brute = bmm.BruteforceSearch(max_candidates=8, chunk_bits=0)

        window = matrix.row_window([0, 1, 2, 3, 4, 5])
        expected = fwht.search(window)
        actual = brute.search(window)

        self.assertEqual(brute.name(), "bruteforce")
        self.assertEqual(
            [(candidate.mask_u64(), candidate.weight) for candidate in actual],
            [(candidate.mask_u64(), candidate.weight) for candidate in expected],
        )

    def test_search_apply(self) -> None:
        matrix = make_search_matrix()
        searcher = bmm.FwhtSearch(max_rows=16, max_candidates=1)
        applier = bmm.GreedyApplier(min_gain=1)

        window = matrix.row_window([0, 1])
        result = bmm.search_apply(
            window,
            searcher=searcher,
            applier=applier,
        )

        self.assertGreaterEqual(result.applied_count, 1)
        self.assertGreater(result.weight_improvement, 0)
        self.assertIn(0, [matrix.row_popcount(0), matrix.row_popcount(1)])

    def test_sa_selector_wrapper(self) -> None:
        matrix = bmm.matrix_from_rows(
            [
                "111111000000",
                "111111000000",
                "111111000000",
                "111111000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
            ]
        )

        selector = bmm.SASelector(iterations=256, restarts=8, seed=7)

        result = selector.select(matrix, 4)
        self.assertEqual(result.rows, [0, 1, 2, 3])
        self.assertEqual(result.score, 72)

        window = selector.select_window(matrix, 4)
        self.assertEqual(window.rows, [0, 1, 2, 3])
        self.assertEqual(window.materialize().to_rows(), ["111111000000"] * 4)
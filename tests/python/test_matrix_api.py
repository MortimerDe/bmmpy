from __future__ import annotations

import unittest

import bmmpy as bmm


class TestMatrixApi(unittest.TestCase):
    def test_bit_matrix_python_sugar(self) -> None:
        matrix = bmm.matrix_from_rows(["10", "01"])

        self.assertTrue(matrix[0, 0])
        self.assertFalse(matrix[0, 1])

        matrix[0, 1] = True
        self.assertEqual(matrix.to_rows(), ["11", "01"])
        self.assertEqual(str(matrix), matrix.to_text())

        copied = matrix.copy()
        copied[0, 0] = False

        self.assertEqual(matrix.to_rows(), ["11", "01"])
        self.assertEqual(copied.to_rows(), ["01", "01"])

        window = matrix.row_window([1, 0])
        self.assertIsInstance(window, bmm.RowWindow)
        self.assertEqual(len(window), 2)
        self.assertEqual(window.cols, 2)
        self.assertEqual(window.rows, [1, 0])
        self.assertEqual(window.materialize().to_rows(), ["01", "11"])

    def test_row_window_python_metadata(self) -> None:
        matrix = bmm.matrix_from_rows([
            "10101",
            "00101",
            "00010",
        ])
        window = matrix.row_window([0, 1, 2])

        self.assertEqual(window.row_popcount(0), 3)
        self.assertEqual(window.row_popcount(1), 2)
        self.assertEqual(window.row_popcount(2), 1)
        self.assertEqual(window.total_weight, 4)

    def test_exact_basis_solver_finds_min_total_weight_basis(self) -> None:
        matrix = bmm.matrix_from_rows([
            "111",
            "110",
            "001",
        ])
        window = matrix.row_window([0, 1, 2])

        result = bmm.solver.ExactBasisSolver().solve(window)

        self.assertEqual(result.input_rows, 3)
        self.assertEqual(result.cols, 3)
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.enumerated_states, 7)
        self.assertEqual(result.total_weight, 3)
        self.assertEqual(sorted(result.basis_weights), [1, 2])
        self.assertEqual(result.transform_matrix.shape, (2, 3))
        self.assertEqual(result.basis_matrix.shape, (2, 3))
        self.assertEqual(result.basis_matrix.rank(), 2)
        self.assertEqual(result.basis_matrix.weight(), 3)

    def test_global_greedy_applier_preserves_rank_and_reduces_weight(self) -> None:
        matrix = bmm.matrix_from_rows([
            "111",
            "110",
            "001",
        ])
        window = matrix.row_window([0, 1, 2])

        candidates = [
            bmm.Candidate.from_u64(0b011, 1),
            bmm.Candidate.from_u64(0b101, 2),
            bmm.Candidate.from_u64(0b110, 3),
        ]

        before_rank = window.materialize().rank()
        before_weight = window.materialize().weight()

        result = bmm.GlobalGreedyApplier().apply(window, candidates)

        self.assertEqual(result.applied_count, 1)
        self.assertGreater(result.weight_improvement, 0)
        self.assertEqual(window.materialize().rank(), before_rank)
        self.assertLess(window.materialize().weight(), before_weight)

    def test_global_greedy_applier_skips_non_improving_identity_basis(self) -> None:
        matrix = bmm.matrix_from_rows([
            "10",
            "01",
        ])
        before = matrix.to_rows()

        window = matrix.row_window([0, 1])
        result = bmm.GlobalGreedyApplier().apply(window, [])

        self.assertEqual(result.applied_count, 0)
        self.assertEqual(result.weight_improvement, 0)
        self.assertEqual(matrix.to_rows(), before)
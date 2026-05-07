from __future__ import annotations

import unittest

import bmmpy as bmm
from bmmpy.algebra.mastrovito.conversions import build_basis_change_matrix, invert_matrix


class TestAlgebraMastrovito(unittest.TestCase):
    def test_algebra_parse_poly(self) -> None:
        expected = (8, [8, 5, 3, 1, 0])

        self.assertEqual(bmm.algebra.parse_poly("x^8 + x^5 + x^3 + x + 1"), expected)
        self.assertEqual(bmm.algebra.parse_poly([8, 5, 3, 1, 0]), expected)
        self.assertEqual(bmm.algebra.parse_poly((8, [8, 5, 3, 1, 0])), expected)

    def test_algebra_mastrovito_blocks(self) -> None:
        mastrovito = bmm.algebra.Mastrovito("x^3 + x + 1")

        self.assertEqual(
            mastrovito.get_mastrovito_matrix(0).to_rows(),
            ["100", "010", "001"],
        )
        self.assertEqual(
            mastrovito.get_mastrovito_matrix(1).to_rows(),
            ["001", "101", "010"],
        )

    def test_algebra_build_check_matrix(self) -> None:
        matrix = bmm.algebra.build_check_matrix("x^3 + x + 1", c=2, k=1, start_i=1)

        self.assertEqual(matrix.shape, (3, 6))
        self.assertEqual(matrix.to_rows(), ["100001", "010101", "001010"])

    def test_algebra_find_row_transform(self) -> None:
        source = bmm.matrix_from_rows([
            "1011",
            "0110",
            "1100",
        ])
        expected_transform = bmm.matrix_from_rows([
            "110",
            "011",
        ])
        target = expected_transform @ source

        actual_transform = bmm.algebra.find_row_transform(source, target)

        self.assertEqual((actual_transform @ source).to_rows(), target.to_rows())

    def test_algebra_find_row_transform_rejects_out_of_span(self) -> None:
        source = bmm.matrix_from_rows([
            "10",
            "10",
        ])
        target = bmm.matrix_from_rows([
            "01",
        ])

        with self.assertRaises(bmm.algebra.RowTransformError):
            bmm.algebra.find_row_transform(source, target)

    def test_algebra_mastrovito_basis_change_matches_conjugation(self) -> None:
        basis = [0, 1, [2, 0]]

        standard = bmm.algebra.Mastrovito("x^3 + x + 1")
        changed = bmm.algebra.Mastrovito("x^3 + x + 1", basis=basis)

        change = build_basis_change_matrix(basis, 3, [3, 1, 0])
        change_inv = invert_matrix(change)

        expected_block = change_inv @ standard.get_mastrovito_matrix(1) @ change

        self.assertEqual(
            changed.get_mastrovito_matrix(1).to_rows(),
            expected_block.to_rows(),
        )
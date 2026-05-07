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
    
    def test_algebra_element_mask_matches_standard_mask_without_basis(self) -> None:
        from_element = bmm.algebra.Mastrovito("x^8 + x^4 + x^3 + x^2 + 1", element=126)
        from_mask = bmm.algebra.Mastrovito("x^8 + x^4 + x^3 + x^2 + 1", element_mask=126)

        self.assertEqual(
            from_element.get_mastrovito_matrix(1).to_rows(),
            from_mask.get_mastrovito_matrix(1).to_rows(),
        )

    def test_algebra_element_mask_matches_custom_basis_coordinates(self) -> None:
        basis = [25, 26, 51, 52, 228, 229, 254, 255]
        poly = "x^8 + x^4 + x^3 + x^2 + 1"

        from_basis_coordinates = bmm.algebra.Mastrovito(
            poly,
            basis=basis,
            element=[26, 51, 52, 228, 229, 254],
        )
        from_mask = bmm.algebra.Mastrovito(
            poly,
            basis=basis,
            element_mask=126,
        )

        self.assertEqual(
            from_basis_coordinates.build_check_matrix(64, 56).hash(),
            from_mask.build_check_matrix(64, 56).hash(),
        )

    def test_algebra_rejects_both_element_and_element_mask(self) -> None:
        with self.assertRaises(ValueError):
            bmm.algebra.Mastrovito(
                "x^3 + x + 1",
                element="x",
                element_mask=0b010,
            )

    def test_algebra_rejects_element_mask_that_exceeds_degree(self) -> None:
        with self.assertRaises(ValueError):
            bmm.algebra.Mastrovito(
                "x^3 + x + 1",
                element_mask=0b1000,
            )
    
    def test_algebra_normalizes_element_representations(self) -> None:
        basis = [25, 26, 51, 52, 228, 229, 254, 255]
        poly = "x^8 + x^4 + x^3 + x^2 + 1"

        from_element = bmm.algebra.Mastrovito(
            poly,
            basis=basis,
            element=[26, 51, 52, 228, 229, 254],
        )
        from_mask = bmm.algebra.Mastrovito(
            poly,
            basis=basis,
            element_mask=126,
        )

        self.assertEqual(from_element.element, [7, 6, 4, 0])
        self.assertEqual(from_mask.element, [7, 6, 4, 0])
        self.assertEqual(from_element.element_mask, 126)
        self.assertEqual(from_mask.element_mask, 126)
        self.assertEqual(from_element.element_standard_mask, 209)
        self.assertEqual(from_mask.element_standard_mask, 209)
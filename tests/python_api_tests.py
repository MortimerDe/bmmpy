from __future__ import annotations

import unittest
import pytest
import os
import bmmpy as bmm

def make_search_matrix() -> bmm.BitMatrix:
    matrix = bmm.BitMatrix(2, 5)
    for col in (0, 2, 4):
        matrix[0, col] = True
        matrix[1, col] = True
    return matrix

def make_cuda_equivalence_matrix(rows: int = 16, cols: int = 512, seed: int = 42) -> bmm.BitMatrix:
    import random
    rng = random.Random(seed)
    matrix = bmm.BitMatrix(rows, cols)

    for row in range(rows):
        for col in range(cols):
            if rng.getrandbits(1):
                matrix[row, col] = True

    return matrix

class PublicApiTests(unittest.TestCase):
    def test_public_api_hides_configs_and_legacy_helpers(self) -> None:
        self.assertFalse(hasattr(bmm, "BruteforceSearchConfig"))
        self.assertFalse(hasattr(bmm, "FwhtSearchConfig"))
        self.assertFalse(hasattr(bmm, "MitmFwhtSearchConfig"))
        self.assertFalse(hasattr(bmm, "make_fwht_config"))
        self.assertFalse(hasattr(bmm, "make_mitm_fwht_config"))
        self.assertFalse(hasattr(bmm, "fwht_search"))
        self.assertFalse(hasattr(bmm, "mitm_fwht_search"))
        self.assertFalse(hasattr(bmm, "apply_greedy"))
        self.assertTrue(hasattr(bmm, "RowWindow"))
        self.assertTrue(hasattr(bmm, "algebra"))
        self.assertFalse(hasattr(bmm, "Mastrovito"))
        self.assertFalse(hasattr(bmm, "build_check_matrix"))
        self.assertFalse(hasattr(bmm, "get_mastrovito_matrix"))
        self.assertFalse(hasattr(bmm, "parse_poly"))
        self.assertFalse(hasattr(bmm, "find_row_transform"))
        self.assertTrue(hasattr(bmm, "CudaBruteforceSearch"))
        self.assertFalse(hasattr(bmm, "CudaBruteforceSearchConfig"))

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
    
    def test_candidate_python_sugar(self) -> None:
        candidate = bmm.Candidate.from_u64(0b1011, 7)

        self.assertEqual(len(candidate), 3)
        self.assertIn(0, candidate)
        self.assertIn(1, candidate)
        self.assertIn(3, candidate)
        self.assertNotIn(2, candidate)
        self.assertEqual(list(candidate), [0, 1, 3])
        self.assertEqual(candidate.rows, [0, 1, 3])
    
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
        mitm = bmm.MitmFwhtSearch(
            max_candidates=8,
        )

        window = matrix.row_window([0, 1, 2, 3, 4, 5])
        expected = fwht.search(window)
        actual = mitm.search(window)

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

    def test_runtime_features_exposes_cuda_flags(self) -> None:
        features = bmm.get_runtime_features()

        self.assertTrue(hasattr(features, "cuda_compiled"))
        self.assertTrue(hasattr(features, "cuda_available"))

        if features.cuda_available:
            self.assertTrue(features.cuda_compiled)
    
    def test_cuda_mitm_search_wrapper_name(self) -> None:
        searcher = bmm.CudaMitmFwhtSearch(max_candidates=8, low_bits=0)
        self.assertEqual(searcher.name(), "cuda_mitm_fwht")

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
    
    def test_cuda_build_expectation_matches_runtime_features(self) -> None:
        expected = os.environ.get("BMMPY_EXPECT_CUDA_COMPILED")
        if expected is None:
            return

        normalized = expected.strip().upper()
        self.assertIn(normalized, {"ON", "OFF"})

        features = bmm.get_runtime_features()
        expected_compiled = normalized == "ON"

        self.assertEqual(
            features.cuda_compiled,
            expected_compiled,
            (
                f"Expected cuda_compiled={expected_compiled} from "
                f"BMMPY_EXPECT_CUDA_COMPILED={normalized}, got {features.cuda_compiled}. "
                "If you changed build flags, reinstall with the matching make target first."
            ),
        )

    def test_cuda_bruteforce_search_wrapper_name(self) -> None:
        searcher = bmm.CudaBruteforceSearch(max_candidates=8, chunk_bits=0)
        self.assertEqual(searcher.name(), "cuda_bruteforce")
    
    def test_cuda_bruteforce_search_wrapper_runtime_behavior(self) -> None:
        features = bmm.get_runtime_features()

        if not features.cuda_compiled:
            pytest.skip("CUDA not compiled into this build")

        if not features.cuda_available:
            pytest.skip("no CUDA device available")

        cpu = bmm.BruteforceSearch(max_candidates=8, chunk_bits=0)
        gpu = bmm.CudaBruteforceSearch(max_candidates=8, chunk_bits=0)
        matrix = make_cuda_equivalence_matrix(rows=16, cols=512)
        window = matrix.row_window(list(range(16)))

        self.assertEqual(
            [(c.mask_u64(), c.weight) for c in gpu.search(window)],
            [(c.mask_u64(), c.weight) for c in cpu.search(window)],
        )
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

if __name__ == "__main__":
    unittest.main()
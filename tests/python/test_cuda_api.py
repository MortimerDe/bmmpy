from __future__ import annotations

import os
import random
import unittest

import pytest

import bmmpy as bmm


def make_cuda_equivalence_matrix(rows: int = 16, cols: int = 512, seed: int = 42) -> bmm.BitMatrix:
    rng = random.Random(seed)
    matrix = bmm.BitMatrix(rows, cols)

    for row in range(rows):
        for col in range(cols):
            if rng.getrandbits(1):
                matrix[row, col] = True

    return matrix


class TestCudaApi(unittest.TestCase):
    def test_runtime_features_exposes_cuda_flags(self) -> None:
        features = bmm.get_runtime_features()

        self.assertTrue(hasattr(features, "cuda_compiled"))
        self.assertTrue(hasattr(features, "cuda_available"))

        if features.cuda_available:
            self.assertTrue(features.cuda_compiled)

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
                "If you changed build flags, rebuild with matching CMake flags."
            ),
        )

    def test_cuda_mitm_search_wrapper_name(self) -> None:
        searcher = bmm.CudaMitmFwhtSearch(max_candidates=8, low_bits=0)
        self.assertEqual(searcher.name(), "cuda_mitm_fwht")

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
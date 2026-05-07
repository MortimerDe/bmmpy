from __future__ import annotations

import unittest

import bmmpy as bmm


class TestPublicApi(unittest.TestCase):
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

    def test_candidate_python_sugar(self) -> None:
        candidate = bmm.Candidate.from_u64(0b1011, 7)

        self.assertEqual(len(candidate), 3)
        self.assertIn(0, candidate)
        self.assertIn(1, candidate)
        self.assertIn(3, candidate)
        self.assertNotIn(2, candidate)
        self.assertEqual(list(candidate), [0, 1, 3])
        self.assertEqual(candidate.rows, [0, 1, 3])
"""Stage 0 baseline testlari (pytest yo'q — plain unittest).

    python3 -m unittest tests.test_baseline -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.baseline_hrr import (
    bind,
    build_obj_matrix,
    cleanup_buggy_matrix,
    cleanup_proper,
    cleanup_proper_matrix,
    encode_fact,
    inv_permute,
    make_world,
    permute,
    random_hrr,
    retrieve,
    similarity,
    unbind,
)


class TestHRRIdentity(unittest.TestCase):
    def test_random_hrr_unit_norm(self):
        rng = np.random.default_rng(0)
        v = random_hrr(1024, rng)
        self.assertEqual(v.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=9)
        self.assertFalse(np.iscomplexobj(v))

    def test_bind_unbind_cleanup_recovers_original(self):
        """d=1024: bind -> unbind -> cleanup asl vektorni yuqori sim bilan topadi."""
        d = 1024
        rng = np.random.default_rng(1)
        a = random_hrr(d, rng)
        b = random_hrr(d, rng)
        codebook = {f"v{i}": random_hrr(d, rng) for i in range(50)}
        codebook["target"] = a

        c = bind(a, b)
        recovered = unbind(c, b)
        name, sim = cleanup_proper(recovered, codebook)

        self.assertEqual(name, "target")
        self.assertGreater(sim, 0.5)
        # xom o'xshashlik ham yuqori bo'lishi kerak
        self.assertGreater(similarity(recovered, a), 0.5)

    def test_permute_roundtrip_exact(self):
        rng = np.random.default_rng(2)
        v = random_hrr(256, rng)
        np.testing.assert_allclose(inv_permute(permute(v, 7), 7), v)

    def test_fact_encode_retrieve_single_fact(self):
        """Bitta faktli xotirada retrieve -> cleanup aniq ishlaydi."""
        fw = make_world(n_facts=8, d=1024, seed=3)
        M = np.sum(np.stack([encode_fact(fw, s, r, o) for s, r, o in fw["facts"]]), axis=0)
        s, r, o = fw["facts"][0]
        cand = retrieve(fw, M, s, r)
        name, _ = cleanup_proper(cand, fw["objects"])
        self.assertEqual(name, o)

    def test_buggy_cleanup_worse_under_load(self):
        """Yuqori yuklamada (C=N=1000, d=512) elementwise xato aniqlikni tushiradi."""
        from src.baseline_hrr import accuracy_matrix

        fw = make_world(n_facts=1000, d=512, seed=0)
        M = np.sum(np.stack([encode_fact(fw, s, r, o) for s, r, o in fw["facts"]]), axis=0)
        names, mat_n = build_obj_matrix(fw["objects"])

        acc_proper, _ = accuracy_matrix(fw, M, names, mat_n, cleanup_proper_matrix)
        acc_buggy, _ = accuracy_matrix(fw, M, names, mat_n, cleanup_buggy_matrix)
        self.assertGreaterEqual(acc_proper, acc_buggy)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for src/sparse_vsa.py (plain unittest, run with python3 -m unittest)."""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.sparse_vsa import (DEFAULT_D, DEFAULT_K, InvertedIndex, bind, bundle,  # noqa: E402
                            cleanup, encode_fact, inv_permute, permute,
                            random_sparse, similarity, sparsity, unbind)


class TestPrimitives(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.a = random_sparse(DEFAULT_D, DEFAULT_K, self.rng)
        self.b = random_sparse(DEFAULT_D, DEFAULT_K, self.rng)

    def test_random_sparse_shape_dtype_count(self):
        self.assertEqual(self.a.shape, (DEFAULT_D,))
        self.assertEqual(self.a.dtype, np.uint8)
        self.assertEqual(sparsity(self.a), DEFAULT_K)
        self.assertTrue(set(np.unique(self.a)).issubset({0, 1}))
        # exactly k ones, no duplicates possible in a binary vector
        self.assertEqual(len(np.flatnonzero(self.a)), DEFAULT_K)

    def test_unrelated_similarity_near_chance(self):
        self.assertLess(similarity(self.a, self.b), 0.05)  # chance is k/d = 0.01

    def test_bind_output_sparse_and_related_to_both(self):
        c = bind(self.a, self.b, self.rng)
        self.assertEqual(sparsity(c), DEFAULT_K)
        # intersection has ~1 position, the rest is filled from the union, so the
        # product is roughly half a and half b
        self.assertGreater(similarity(c, self.a), 0.3)
        self.assertGreater(similarity(c, self.b), 0.3)

    def test_unbind_roundtrip_beats_chance(self):
        rec = unbind(bind(self.a, self.b, self.rng), self.b, self.rng)
        self.assertEqual(sparsity(rec), DEFAULT_K)
        self.assertGreater(similarity(rec, self.a), 0.1)  # measured ~0.25-0.30
        self.assertGreater(similarity(rec, self.a), 5 * DEFAULT_K / DEFAULT_D)

    def test_permute_is_involutive(self):
        self.assertTrue(np.array_equal(inv_permute(permute(self.a, 1), 1), self.a))

    def test_bind_requires_matching_dims(self):
        with self.assertRaises(ValueError):  # numpy broadcast error path
            bind(self.a, self.a[:100], self.rng)


class TestBundle(unittest.TestCase):
    def test_top_k_by_overlap_count(self):
        rng = np.random.default_rng(1)
        vecs = [random_sparse(500, 20, rng) for _ in range(30)]
        k_out = 50
        m = bundle(vecs, k_out=k_out, rng=rng)
        self.assertEqual(sparsity(m), k_out)
        counts = np.bincount(np.concatenate([np.flatnonzero(v) for v in vecs]),
                             minlength=500)
        chosen = np.flatnonzero(m)
        rest = np.setdiff1d(np.arange(500), chosen)
        # no chosen position may have fewer votes than an unchosen one
        self.assertGreaterEqual(counts[chosen].min(), counts[rest].max())

    def test_default_k_out_is_input_sparsity(self):
        rng = np.random.default_rng(2)
        vecs = [random_sparse(1000, 10, rng) for _ in range(5)]
        self.assertEqual(sparsity(bundle(vecs, rng=rng)), 10)

    def test_bundle_more_similar_to_members_than_to_strangers(self):
        rng = np.random.default_rng(3)
        vecs = [random_sparse(2000, 20, rng) for _ in range(5)]
        m = bundle(vecs, rng=rng)
        stranger = random_sparse(2000, 20, rng)
        self.assertGreater(np.mean([similarity(m, v) for v in vecs]),
                           similarity(m, stranger))


class TestCleanup(unittest.TestCase):
    def test_inverted_index_matches_brute_force(self):
        rng = np.random.default_rng(4)
        d, k, c = 500, 20, 40
        codebook = {f"o{i}": random_sparse(d, k, rng) for i in range(c)}
        index = InvertedIndex(codebook)
        names = list(codebook)
        mat = np.stack([codebook[n] for n in names])
        for _ in range(5):
            cand = random_sparse(d, k, rng)
            true_overlaps = mat @ cand.astype(np.int64)
            name, sim = index.query(cand, k=k)
            i = int(np.argmax(true_overlaps))
            self.assertEqual(name, names[i])
            self.assertAlmostEqual(sim, true_overlaps[i] / k)

    def test_single_fact_roundtrip_and_cleanup(self):
        rng = np.random.default_rng(5)
        d, k = 2000, 20
        s = random_sparse(d, k, rng)
        r = random_sparse(d, k, rng)
        objs = {f"o{i}": random_sparse(d, k, rng) for i in range(10)}
        o = "o7"
        fact = encode_fact(s, r, objs[o], perm=1, rng=rng)
        self.assertEqual(sparsity(fact), k)
        key = bind(s, r, rng)
        cand = inv_permute(unbind(fact, key, rng), 1)
        name, sim = cleanup(cand, objs)
        self.assertEqual(name, o)
        self.assertGreater(sim, 0.1)


if __name__ == "__main__":
    unittest.main()

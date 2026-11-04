"""Unit tests for src/resonator.py (plain unittest, run with python3 -m unittest)."""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.fhrr import DTYPE, bind, inv_permute, permute, random_fhrr, similarity, unbind  # noqa: E402
from src.resonator import (bundle, factorize_2, factorize_3, make_cleanup_fn,  # noqa: E402
                           make_soft_cleanup_fn, pair, triple)

D = 2048
C = 100


def _codebook(prefix, rng, n=C):
    return {f"{prefix}{i}": random_fhrr(D, rng) for i in range(n)}


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.A = _codebook("a", self.rng)
        self.B = _codebook("b", self.rng)

    def test_pair_roles_are_recoverable(self):
        a, b = self.A["a1"], self.B["b2"]
        P = pair(a, b)
        a_est = inv_permute(unbind(P, permute(b, 2)), 1)
        b_est = inv_permute(unbind(P, permute(a, 1)), 2)
        self.assertGreater(similarity(a_est, a), 0.99)
        self.assertGreater(similarity(b_est, b), 0.99)

    def test_triple_roles_are_recoverable(self):
        rng = self.rng
        Cb = _codebook("c", rng)
        a, b, c = self.A["a3"], self.B["b4"], Cb["c5"]
        T = triple(a, b, c)
        a_est = inv_permute(unbind(T, bind(permute(b, 2), permute(c, 3))), 1)
        self.assertGreater(similarity(a_est, a), 0.99)

    def test_cleanup_fns_return_codebook_vectors(self):
        ca = make_cleanup_fn(self.A)
        name, vec = ca(self.A["a7"])
        self.assertEqual(name, "a7")
        self.assertEqual(vec.dtype, DTYPE)
        ca_soft = make_soft_cleanup_fn(self.A)
        name_s, state = ca_soft(self.A["a7"])
        self.assertEqual(name_s, "a7")
        self.assertEqual(state.shape, (D,))


class TestFactorize2(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.A = _codebook("a", self.rng)
        self.B = _codebook("b", self.rng)
        self.ca = make_soft_cleanup_fn(self.A)
        self.cb = make_soft_cleanup_fn(self.B)
        self.ca_hard = make_cleanup_fn(self.A)
        self.cb_hard = make_cleanup_fn(self.B)
        self.a0 = np.mean(list(self.A.values()), axis=0).astype(DTYPE)
        self.b0 = np.mean(list(self.B.values()), axis=0).astype(DTYPE)

    def test_recovers_two_args_isolated_memory(self):
        ok, iters = 0, []
        for t in range(50):
            an = f"a{int(self.rng.integers(C))}"
            bn = f"b{int(self.rng.integers(C))}"
            P = pair(self.A[an], self.B[bn])
            av, bv, ni, converged = factorize_2(P, None, self.a0, self.b0,
                                               self.ca, 1, 2, 10,
                                               cleanup_b=self.cb)
            ok += int(self.ca(av)[0] == an and self.cb(bv)[0] == bn)
            iters.append(ni)
            self.assertTrue(converged)
            self.assertLessEqual(ni, 10)
        self.assertGreater(ok / 50, 0.8)  # measured 1.0, ~2 iterations

    def test_bundled_memory_with_cue_is_fine_at_small_n(self):
        N = 3
        S = _codebook("s", self.rng, n=N)
        facts = [("a1", "b2"), ("a3", "b4"), ("a5", "b6")]
        M = bundle([bind(S[f"s{i}"], pair(self.A[a], self.B[b]))
                    for i, (a, b) in enumerate(facts)])
        ok = 0
        for i, (an, bn) in enumerate(facts):
            av, bv, _ni, _cv = factorize_2(M, S[f"s{i}"], self.a0, self.b0,
                                          self.ca, 1, 2, 10,
                                          cleanup_b=self.cb)
            ok += int(self.ca(av)[0] == an and self.cb(bv)[0] == bn)
        self.assertEqual(ok, N)

    def test_hard_cleanup_also_works_with_uniform_prior(self):
        an, bn = "a8", "b9"
        P = pair(self.A[an], self.B[bn])
        av, bv, _ni, _cv = factorize_2(P, None, self.a0, self.b0,
                                      self.ca_hard, 1, 2, 10,
                                      cleanup_b=self.cb_hard)
        self.assertEqual(self.ca_hard(av)[0], an)
        self.assertEqual(self.cb_hard(bv)[0], bn)


class TestFactorize3(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.A = _codebook("a", self.rng)
        self.B = _codebook("b", self.rng)

    def test_cued_triple_recovers_pair(self):
        rng = self.rng
        Cb = _codebook("c", rng)
        ca, cb = (make_soft_cleanup_fn(self.A),
                  make_soft_cleanup_fn(self.B))
        an, bn, cn = "a10", "b11", "c12"
        T = triple(self.A[an], self.B[bn], Cb[cn])
        M = unbind(T, permute(Cb[cn], 3))  # c given
        a0 = np.mean(list(self.A.values()), axis=0).astype(DTYPE)
        b0 = np.mean(list(self.B.values()), axis=0).astype(DTYPE)
        av, bv, ni, converged = factorize_2(M, None, a0, b0, ca, 1, 2, 10,
                                           cleanup_b=cb)
        self.assertEqual(ca(av)[0], an)
        self.assertEqual(cb(bv)[0], bn)
        self.assertGreaterEqual(ni, 1)


if __name__ == "__main__":
    unittest.main()

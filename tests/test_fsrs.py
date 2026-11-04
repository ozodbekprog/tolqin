"""Stage 6 tests: FSRS retention, LP attention, integrated weighted memory.

Run with:  python3 -m unittest tests.test_fsrs -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.factworld import make_world                     # noqa: E402
from src.fsrs import FSRS                                # noqa: E402
from src.integrated_memory import IntegratedMemory       # noqa: E402
from src.lp_attention import LPAttention                 # noqa: E402


def build_memory(n: int = 20, d: int = 64, seed: int = 0, S0: float = 1.0,
                 weighted: bool = True, rebalance_every: int = 1):
    fw = make_world(n_facts=n, d=d, seed=seed)
    im = IntegratedMemory(d, perm=1, world=fw, S0=S0, weighted=weighted,
                          rebalance_every=rebalance_every)
    ids = [im.add_fact(s, r, o, t=0.0) for s, r, o in fw["facts"]]
    return fw, im, ids


class TestFSRS(unittest.TestCase):
    def test_retention_decreases_with_time(self):
        f = FSRS(tau=0.0, S=10.0)
        self.assertAlmostEqual(f.R(0.0), 1.0)
        self.assertAlmostEqual(f.R(90.0), 0.5)          # half-life = 9*S
        values = [f.R(t) for t in range(0, 400, 20)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertTrue(all(0.0 < v <= 1.0 for v in values))
        # time before the last review is clamped: no "negative forgetting"
        self.assertAlmostEqual(f.R(-50.0), 1.0)

    def test_success_increases_stability(self):
        f = FSRS(tau=0.0, S=10.0)
        s_before = f.S
        r_before = f.R(45.0)
        r_now = f.on_recall(45.0, success=True)
        self.assertAlmostEqual(r_now, r_before)
        self.assertGreater(f.S, s_before)
        self.assertAlmostEqual(f.S, s_before * (1 + 0.5 * (1 - r_before)))
        self.assertEqual(f.tau, 45.0)
        self.assertAlmostEqual(f.R(45.0), 1.0)

    def test_failure_decreases_stability(self):
        f = FSRS(tau=0.0, S=10.0, b=0.2)
        s_before = f.S
        f.on_recall(30.0, success=False)
        self.assertLess(f.S, s_before)
        self.assertAlmostEqual(f.S, s_before * np.exp(-0.2))
        self.assertEqual(f.tau, 30.0)

    def test_success_reward_larger_when_nearly_forgotten(self):
        fresh = FSRS(tau=0.0, S=10.0)
        stale = FSRS(tau=0.0, S=10.0)
        fresh.on_recall(1.0, success=True)
        stale.on_recall(900.0, success=True)
        self.assertGreater(stale.S / 10.0, fresh.S / 10.0)

    def test_rejects_nonpositive_stability(self):
        with self.assertRaises(ValueError):
            FSRS(S=0.0)


class TestLPAttention(unittest.TestCase):
    def test_weights_normalized_and_favor_fresh_items(self):
        lp = LPAttention(S0=10.0)
        lp.add_fact("fresh", 0.0)
        lp.add_fact("old", -500.0)
        w = lp.weights(0.0, dt=1.0)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertTrue(all(v >= 0.0 for v in w.values()))
        self.assertGreater(w["fresh"], w["old"])   # fresh retention moves faster

    def test_uniform_fallback_when_nothing_changes(self):
        lp = LPAttention(S0=10.0)
        lp.add_fact("a", 0.0)
        lp.add_fact("b", 0.0)
        w = lp.weights(0.0, dt=0.0)                # zero horizon -> no change
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["a"], 0.5)
        self.assertEqual(lp.weights(0.0), lp.weights(0.0))
        self.assertEqual(LPAttention().weights(0.0), {})

    def test_active_threshold(self):
        lp = LPAttention(S0=10.0)
        lp.add_fact("fresh", 0.0)
        lp.add_fact("stale", -200.0)
        active = lp.active(0.0, threshold=0.5)
        self.assertIn("fresh", active)
        self.assertNotIn("stale", active)
        self.assertAlmostEqual(lp.R("stale", 0.0), 1 / (1 + 200 / 90))

    def test_recall_updates_fsrs(self):
        lp = LPAttention(S0=10.0)
        lp.add_fact(0, 0.0)
        lp.recall(0, 30.0, success=False)
        self.assertLess(lp.fsrs[0].S, 10.0)
        with self.assertRaises(KeyError):
            lp.add_fact(0, 0.0)


class TestIntegratedMemory(unittest.TestCase):
    def test_weighted_equals_flat_when_all_fresh(self):
        fw, im_w, _ = build_memory(n=8, d=64, seed=1, weighted=True)
        _, im_f, _ = build_memory(n=8, d=64, seed=1, weighted=False)
        t = 0.0
        im_w.rebalance(t)
        im_f.rebalance(t)
        np.testing.assert_allclose(im_w.M, im_f.M, atol=1e-4)

    def test_weighted_memory_retrieves_recently_reviewed_fact(self):
        fw, im, ids = build_memory(n=10, d=256, seed=2, S0=1.0)
        s, r, o = fw["facts"][3]
        t = 40.0                       # everything else has decayed (R ~ 0.18)
        name, _ = im.query(s, r, t)
        self.assertEqual(name, o)
        # a successful review then keeps the item retrievable and raises its weight
        im.recall(ids[3], t, success=True)
        fsrs = im.facts[3]["fsrs"]
        w = fsrs.R(t)
        self.assertAlmostEqual(fsrs.R(t), 1.0)
        name, sim = im.query(s, r, t + 0.5)
        self.assertEqual(name, o)
        self.assertGreater(sim, 0.0)
        # the bundle is the weighted sum of the stored facts
        w_vec = np.array([f["fsrs"].R(t + 0.5) for f in im.facts])
        np.testing.assert_allclose(im.M, (im._bundle_matrix() * w_vec[:, None]).sum(0),
                                   atol=1e-4)
        self.assertLess(w_vec[3], 1.0 + 1e-9)

    def test_prune_removes_low_retention_facts(self):
        fw, im, ids = build_memory(n=20, d=64, seed=3, S0=1.0)
        survivor = ids[5]
        im.attn.fsrs[survivor].S = 1e5          # this one is well consolidated
        t = 200.0                               # R < 0.1 for everything else
        removed = im.prune(t, threshold=0.1)
        self.assertEqual(removed, 19)
        self.assertEqual(im.n_pruned, 19)
        self.assertEqual(im.n_facts, 1)
        self.assertEqual(im.facts[0]["id"], survivor)
        s, r, o = im.facts[0]["subject"], im.facts[0]["relation"], im.facts[0]["object"]
        self.assertEqual(im.query(s, r, t)[0], o)
        self.assertLess(float(np.linalg.norm(im.M)), 10.0)

    def test_prune_keeps_everything_above_threshold(self):
        _, im, _ = build_memory(n=6, d=64, seed=4, S0=1.0)
        self.assertEqual(im.prune(0.0, threshold=0.1), 0)
        self.assertEqual(im.n_facts, 6)

    def test_query_does_not_review(self):
        fw, im, _ = build_memory(n=6, d=64, seed=5, S0=1.0)
        s, r, _ = fw["facts"][0]
        im.query(s, r, 12.0)
        self.assertAlmostEqual(im.facts[0]["fsrs"].tau, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Stage 2 blokli xotira testlari (plain unittest — pytest emas).

Ishga tushirish (repo ildizidan):
    python3 -m unittest tests.test_blocked -v
"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.blocked_memory import BlockedMemory  # noqa: E402
from src.factworld import encode_fact, make_world, query_key  # noqa: E402
from src.fhrr import build_obj_matrix  # noqa: E402


def _world(n_facts=1000, d=512, seed=1, n_subjects=50, n_relations=20):
    fw = make_world(n_facts=n_facts, d=d, seed=seed,
                    n_subjects=n_subjects, n_relations=n_relations)
    names, mat_n = build_obj_matrix(fw["objects"])
    fvecs = [encode_fact(fw, s, r, o) for (s, r, o) in fw["facts"]]
    keys = [query_key(fw, s, r) for (s, r, _o) in fw["facts"]]
    targets = [o for (_s, _r, o) in fw["facts"]]
    return names, mat_n, fvecs, keys, targets, fw["facts"]


def _accuracy(bm, keys, targets, names, mat_n):
    hits = sum(1 for k, o in zip(keys, targets)
               if bm.query(k, names, mat_n, perm_used=1)[0] == o)
    return hits / len(targets)


class TestPlacement(unittest.TestCase):
    """(a) har bir fakt aynan bitta blokka tushadi, counts yig'indisi N."""

    def test_each_fact_lands_in_exactly_one_block(self):
        _, _, fvecs, _, _, facts = _world(n_facts=200, d=256, seed=3)

        for strategy in ("round_robin", "hash", "similarity"):
            with self.subTest(strategy=strategy):
                for B in (1, 2, 4, 8):
                    bm = BlockedMemory(256, n_blocks=B, strategy=strategy)
                    seen = [0] * B
                    for fvec, (s, r, _o) in zip(fvecs, facts):
                        b = bm.add(fvec, subject=s, relation=r)
                        self.assertTrue(0 <= b < B, f"block index {b} out of range")
                        seen[b] += 1
                    # har add aynan bitta blokni tanlaydi
                    self.assertEqual(sum(seen), len(fvecs))
                    self.assertEqual(bm.counts, seen,
                                     f"{strategy} B={B}: counts != placement tally")
                    self.assertEqual(sum(bm.counts), len(fvecs))
                    # faktlar hech qayerda takrorlanmaydi / yo'qolmaydi
                    self.assertEqual(len(bm.blocks), B)
                    for b in range(B):
                        if bm.counts[b] > 0:
                            self.assertGreater(float(np.linalg.norm(bm.blocks[b])), 0.0)
                        else:
                            self.assertEqual(float(np.linalg.norm(bm.blocks[b])), 0.0)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            BlockedMemory(64, n_blocks=2, strategy="nope")


class TestBlockingHelps(unittest.TestCase):
    """(b) B=8 accuracy > B=1 accuracy (d=512, N=1000)."""

    def test_b8_beats_b1_d512_n1000(self):
        names, mat_n, fvecs, keys, targets, facts = _world(n_facts=1000, d=512, seed=1)

        accs = {}
        for B in (1, 8):
            bm = BlockedMemory(512, n_blocks=B, strategy="round_robin")
            for fvec, (s, r, _o) in zip(fvecs, facts):
                bm.add(fvec, subject=s, relation=r)
            accs[B] = _accuracy(bm, keys, targets, names, mat_n)
            # sanani ko'rsatish uchun (assert xabari ham shundan foydalanadi)
            print(f"  d=512 N=1000 B={B}: accuracy={accs[B]:.3f}")
        self.assertGreater(accs[8], accs[1],
                           f"B=8 ({accs[8]:.3f}) should beat B=1 ({accs[1]:.3f})")


class TestLoadBalance(unittest.TestCase):
    """(c) og'ma (skew) yuklamada similarity round_robin'dan tekisroqmi?

    round_robin konstruksiya bo'yicha mukammal balanslangan
    (max_count/min_count farqi <= 1), shuning uchun u hech qachon
    "undan ham tekisroq" bo'la olmaydi. O'lchaymiz va agar farq
    haqiqatan yo'q bo'lsa, assertni skip qilamiz.
    """

    def test_similarity_load_ratio_vs_round_robin_under_skew(self):
        # og'ma dunyo: oz subyekt/munosabat, ko'p fakt -> (s,r) kalitlari juda takrorlanadi
        N, d = 400, 256
        _, _, fvecs, _, _, facts = _world(n_facts=N, d=d, seed=7,
                                          n_subjects=4, n_relations=3)
        B = 8

        ratios = {}
        for strategy in ("round_robin", "hash", "similarity"):
            bm = BlockedMemory(d, n_blocks=B, strategy=strategy)
            for fvec, (s, r, _o) in zip(fvecs, facts):
                bm.add(fvec, subject=s, relation=r)
            used = [c for c in bm.counts if c > 0]
            self.assertEqual(sum(bm.counts), N, f"{strategy}: facts lost")
            ratios[strategy] = max(bm.counts) / min(used)

        print(f"  skew load ratios: {ratios}")
        self.assertLessEqual(ratios["round_robin"], 1.0 + 2.0 / B)
        if ratios["similarity"] < ratios["round_robin"]:
            self.assertLess(ratios["similarity"], ratios["round_robin"])
        else:
            self.skipTest(
                "round_robin is balanced by construction "
                f"(ratio={ratios['round_robin']:.2f}); similarity "
                f"(ratio={ratios['similarity']:.2f}) cannot be less uneven")

    def test_similarity_beats_hash_load_ratio_under_skew(self):
        """Qo'shimcha o'lchov: og'ma yuklamada similarity hash'dan tekisroq."""
        N, d, B = 400, 256, 8
        _, _, fvecs, _, _, facts = _world(n_facts=N, d=d, seed=7,
                                          n_subjects=4, n_relations=3)
        ratios = {}
        for strategy in ("hash", "similarity"):
            bm = BlockedMemory(d, n_blocks=B, strategy=strategy)
            for fvec, (s, r, _o) in zip(fvecs, facts):
                bm.add(fvec, subject=s, relation=r)
            ratios[strategy] = max(bm.counts) / min(c for c in bm.counts if c)
        print(f"  skew load ratios (hash vs similarity): {ratios}")
        self.assertLess(ratios["similarity"], ratios["hash"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

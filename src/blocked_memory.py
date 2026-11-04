"""Blokli xotira + load balancing strategiyalari.

MUZLATILGAN: faqat asosiy agent o'zgartiradi.
"""
from __future__ import annotations

from zlib import crc32

import numpy as np

from src.fhrr import DTYPE, cleanup_matrix, inv_permute, similarity, unbind

STRATEGIES = ("round_robin", "hash", "similarity")


class BlockedMemory:
    def __init__(self, d: int, n_blocks: int = 1, strategy: str = "round_robin",
                 perm: int = 1, seed: int = 0):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy {strategy!r} not in {STRATEGIES}")
        self.d = d
        self.B = n_blocks
        self.strategy = strategy
        self.perm = perm
        self.blocks = [np.zeros(d, dtype=DTYPE) for _ in range(n_blocks)]
        self.counts = [0] * n_blocks
        self._rr = 0
        self._rng = np.random.default_rng(seed)

    def _choose_block(self, fvec: np.ndarray, subject, relation) -> int:
        if self.B == 1:
            return 0
        if self.strategy == "round_robin":
            b = self._rr
            self._rr = (self._rr + 1) % self.B
            return b
        if self.strategy == "hash":
            return crc32(f"{subject}|{relation}".encode()) % self.B
        # similarity: eng kam o'xshash blok
        sims = [abs(similarity(m, fvec)) if c else -1.0
                for m, c in zip(self.blocks, self.counts)]
        return int(np.argmin(sims))

    def add(self, fvec: np.ndarray, subject=None, relation=None) -> int:
        b = self._choose_block(fvec, subject, relation)
        self.blocks[b] = (self.blocks[b] + fvec).astype(DTYPE)
        self.counts[b] += 1
        return b

    def _block_proxy(self, key: np.ndarray) -> np.ndarray:
        """Arzon proxy: blok energiyasi bilan bog'langan o'xshashlik."""
        return np.array([abs(similarity(m, key)) if c else -1.0
                         for m, c in zip(self.blocks, self.counts)])

    def query(self, key: np.ndarray, names: list[str], mat_n: np.ndarray,
              perm_used: int = 1, top_k: int | None = None,
              early_stop: float | None = None):
        """Returns (name, sim, n_blocks_scanned)."""
        order = list(range(self.B))
        if top_k is not None and top_k < self.B:
            proxy = self._block_proxy(key)
            order = list(np.argsort(proxy)[::-1][:top_k])
        best_name, best_sim = None, -2.0
        scanned = 0
        for b in order:
            if self.counts[b] == 0:
                continue
            cand = inv_permute(unbind(self.blocks[b], key), self.perm)
            name, sim, _ = cleanup_matrix(cand, names, mat_n)
            scanned += 1
            if sim > best_sim:
                best_name, best_sim = name, sim
            if early_stop is not None and best_sim > early_stop:
                break
        if best_name is None:
            return names[0], -1.0, 0
        return best_name, best_sim, scanned

    def memory_bytes(self) -> int:
        return sum(m.nbytes for m in self.blocks)

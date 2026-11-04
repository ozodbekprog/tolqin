"""Integrated memory: one FHRR bundle whose item weights are FSRS retentions.

    M = sum_i w_i * f_i,   w_i = R_i(t) in (0, 1]

Weights are not per-query quantities (that would cost a full rebuild per
query); they are refreshed lazily: on the first query after a structural
change (add / prune) and every ``rebalance_every`` queries. A review only
changes FSRS state, so it becomes visible at the next rebalance; the lag is
bounded by ``rebalance_every`` queries. Retrieval always runs against the
single dense bundle with ``cleanup_matrix`` over the object codebook.

``weighted=False`` builds the same bundle with all ``w_i = 1`` (the flat
baseline used by the stage-6 comparison).

Encoding follows the frozen contract: ``f = bind(bind(S, R), permute(O))`` and
retrieval is ``inv_permute(unbind(unbind(M, S), R))``. Subject / relation /
object codebooks are taken from the same world dict that produced the facts.
"""
from __future__ import annotations

import numpy as np

from .factworld import encode_fact, retrieve
from .fhrr import DTYPE, build_obj_matrix, cleanup_matrix
from .lp_attention import LPAttention


class IntegratedMemory:
    def __init__(self, d: int, perm: int = 1, objects: dict | None = None,
                 world: dict | None = None, subjects: dict | None = None,
                 relations: dict | None = None, a: float = 0.5, b: float = 0.2,
                 S0: float = 1.0, rebalance_every: int = 50,
                 weighted: bool = True):
        if world is not None:
            objects = world["objects"] if objects is None else objects
            subjects = world["subjects"] if subjects is None else subjects
            relations = world["relations"] if relations is None else relations
        self.d = int(d)
        self.perm = int(perm)
        self.weighted = bool(weighted)
        self.rebalance_every = max(1, int(rebalance_every))
        self.subjects = subjects or {}
        self.relations = relations or {}

        self.attn = LPAttention(a=a, b=b, S0=S0)
        self.facts: list[dict] = []      # {"id","subject","relation","object","fsrs"}
        self._vecs: list[np.ndarray] = []  # aligned with self.facts
        self._Fmat: np.ndarray | None = None

        self.M = np.zeros(self.d, dtype=DTYPE)
        self._dirty = True                # bundle out of date
        self._queries_since_rebalance = 0
        self._next_id = 0
        self.n_pruned = 0
        self.n_rebalances = 0
        self.last_rebalance_t = None

        self._obj_names: list[str] | None = None
        self._obj_mat: np.ndarray | None = None
        self._objects = objects

    # ------------------------------------------------------------------ add
    def add_fact(self, s, r, o, t: float, fvec: np.ndarray | None = None) -> int:
        """Add one fact with first review at time t. Returns the fact id."""
        if fvec is None:
            if o not in (self._objects or {}):
                raise ValueError("add_fact needs fvec or the object in the codebook")
            if s not in self.subjects or r not in self.relations:
                raise ValueError("add_fact needs fvec or subject/relation codebooks")
            fvec = encode_fact({"subjects": self.subjects, "relations": self.relations,
                                "objects": self._objects}, s, r, o, self.perm)
        fid = self._next_id
        self._next_id += 1
        fsrs = self.attn.add_fact(fid, t)
        self.facts.append({"id": fid, "subject": s, "relation": r,
                           "object": o, "fsrs": fsrs})
        self._vecs.append(np.asarray(fvec, dtype=DTYPE))
        self._Fmat = None
        self._dirty = True
        return fid

    # ---------------------------------------------------------------- bundle
    def _bundle_matrix(self) -> np.ndarray:
        if self._Fmat is None:
            self._Fmat = (np.stack(self._vecs) if self._vecs
                          else np.zeros((0, self.d), dtype=DTYPE))
        return self._Fmat

    def rebalance(self, t: float) -> np.ndarray:
        """Recompute M = sum_i w_i f_i with w_i = R_i(t) (or 1 if flat)."""
        if not self.facts:
            self.M = np.zeros(self.d, dtype=DTYPE)
        else:
            if self.weighted:
                w = np.fromiter((f["fsrs"].R(t) for f in self.facts),
                                dtype=np.float32, count=len(self.facts))
            else:
                w = np.ones(len(self.facts), dtype=np.float32)
            self.M = (self._bundle_matrix() * w[:, None]).sum(axis=0).astype(DTYPE)
        self._dirty = False
        self._queries_since_rebalance = 0
        self.last_rebalance_t = float(t)
        self.n_rebalances += 1
        return self.M

    def _maybe_rebalance(self, t: float) -> None:
        if self._dirty or self._queries_since_rebalance >= self.rebalance_every:
            self.rebalance(t)

    # ---------------------------------------------------------------- query
    def _codebook(self) -> tuple[list[str], np.ndarray]:
        if self._objects is None:
            raise ValueError("IntegratedMemory needs an objects codebook for cleanup")
        if self._obj_names is None:
            self._obj_names, self._obj_mat = build_obj_matrix(self._objects)
        return self._obj_names, self._obj_mat

    def query(self, s, r, t: float) -> tuple[str, float]:
        """Clean up the object for key (s, r) at time t. Does not touch FSRS."""
        if s not in self.subjects or r not in self.relations:
            raise KeyError(f"unknown subject/relation ({s!r}, {r!r})")
        self._maybe_rebalance(t)
        self._queries_since_rebalance += 1
        names, mat = self._codebook()
        cand = retrieve({"subjects": self.subjects, "relations": self.relations},
                        self.M, s, r, self.perm)
        name, sim, _ = cleanup_matrix(cand, names, mat)
        return name, sim

    def recall(self, fid: int, t: float, success: bool = True) -> float:
        """Review one fact (by id): updates its FSRS; M catches up at rebalance."""
        r_now = self.attn.recall(fid, t, success)
        return r_now

    # ---------------------------------------------------------------- prune
    def prune(self, t: float, threshold: float = 0.1) -> int:
        """Drop facts with R(t) < threshold; rebuild M without them."""
        keep = [i for i, f in enumerate(self.facts)
                if self.attn.R(f["id"], t) >= threshold]
        removed = len(self.facts) - len(keep)
        if removed == 0:
            return 0
        kept = set(keep)
        for i, f in enumerate(self.facts):
            if i not in kept:
                self.attn.drop_fact(f["id"])
        self.facts = [self.facts[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self.n_pruned += removed
        self._Fmat = None
        self._dirty = True
        self.rebalance(t)
        return removed

    # ------------------------------------------------------------- retention
    def mean_R(self, t: float) -> float:
        if not self.facts:
            return 0.0
        return float(np.mean([f.R(t) for f in self.attn.fsrs.values()]))

    def fraction_retained(self, t: float, threshold: float = 0.5) -> float:
        """Share of still-stored facts with R(t) > threshold."""
        if not self.facts:
            return 0.0
        return sum(1 for f in self.attn.fsrs.values() if f.R(t) > threshold) / len(self.facts)

    @property
    def n_facts(self) -> int:
        return len(self.facts)

    def active(self, t: float, threshold: float = 0.5) -> list:
        return self.attn.active(t, threshold)

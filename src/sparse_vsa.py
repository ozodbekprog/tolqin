"""Sparse binary VSA (SBC / Frady 2018 style), d=10000, k=100 defaults.

Vectors are ``np.uint8`` arrays of length ``d`` with exactly ``k`` ones.
All operations are deterministic given the ``rng`` argument.

Binding rule (context-dependent thinning, exact rule)
-----------------------------------------------------
``bind(a, b)``:
  1. ``inter = np.flatnonzero(a & b)``.
  2. If ``len(inter) >= k`` (k = number of ones in ``a``): keep ``k`` positions
     sampled uniformly without replacement from ``inter``.  (If exactly ``k``,
     keep all of them.)
  3. Else keep all of ``inter`` and fill up to ``k`` with positions sampled
     uniformly without replacement from ``union(a, b) \\ inter`` (i.e. from
     ``a \\ b`` and ``b \\ a``).
  4. If ``union`` itself has fewer than ``k`` positions (near-identical
     inputs), take the whole union and fill the remainder from the complement
     of the union, again uniformly at random.

``unbind(c, b) = bind(c, b)``: thinning makes binding only *approximately*
self-inverse.  For independent random vectors the intersection of a product
with either factor is tiny (k^2/d ~ 1), so the filler positions are mostly
noise.  Unbinding therefore works only in combination with bundling and
cleanup, and it degrades faster with the number of stored facts than dense
FHRR.  This is measured, not assumed, in ``scripts/stage4_dense_vs_sparse.py``.

Bundling rule
-------------
``bundle(vecs, k_out=None)``: count how many input vectors have a one at each
position (``np.bincount`` over the concatenated ``flatnonzero`` indices) and
keep the ``k_out`` positions with the highest count (``argpartition``).
Ties are broken uniformly at random (small random jitter added to the counts
before partitioning).  ``k_out`` defaults to the sparsity of the inputs.

Similarity
----------
``similarity(a, b)`` = ``|a & b| / k`` with ``k`` the sparsity of ``a``.
Expected value for two unrelated sparse vectors is ``k / d = 0.01``.

Fact convention (matches src/factworld.py)
------------------------------------------
``encode_fact(s, r, o, perm=1) = bind(bind(s, r), permute(o, perm))``.
"""
from __future__ import annotations

import numpy as np

SPARSE_DTYPE = np.uint8
DEFAULT_D = 10000
DEFAULT_K = 100


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def random_sparse(d: int = DEFAULT_D, k: int = DEFAULT_K,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Sparse binary vector: d positions, exactly k ones."""
    rng = np.random.default_rng() if rng is None else rng
    if not 0 <= k <= d:
        raise ValueError(f"need 0 <= k <= d, got k={k}, d={d}")
    v = np.zeros(d, dtype=SPARSE_DTYPE)
    if k:
        v[rng.choice(d, size=k, replace=False)] = 1
    return v


def sparsity(v: np.ndarray) -> int:
    return int(np.count_nonzero(v))


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    """Cyclic shift by k (same convention as src.fhrr.permute)."""
    return np.roll(v, k)


def inv_permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(v, -k)


def bind(a: np.ndarray, b: np.ndarray,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """Context-dependent thinning bind.  See module docstring for exact rule."""
    rng = np.random.default_rng() if rng is None else rng
    k = sparsity(a)
    inter = np.flatnonzero(a & b)
    if len(inter) >= k:
        chosen = inter if len(inter) == k else rng.choice(inter, size=k, replace=False)
    else:
        union = np.flatnonzero(a | b)
        rest = union[~np.isin(union, inter)]
        need = k - len(inter)
        if len(rest) >= need:
            chosen = np.concatenate([inter, rng.choice(rest, size=need, replace=False)])
        else:
            chosen = union
            extra = k - len(chosen)
            if extra > 0:
                comp = np.setdiff1d(np.arange(len(a)), union, assume_unique=True)
                chosen = np.concatenate(
                    [chosen, rng.choice(comp, size=extra, replace=False)])
    out = np.zeros_like(a, dtype=SPARSE_DTYPE)
    out[chosen] = 1
    return out


def unbind(c: np.ndarray, b: np.ndarray,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Approximate inverse of bind; identical rule to bind (documented above)."""
    return bind(c, b, rng=rng)


def bundle(vecs: list[np.ndarray], k_out: int | None = None,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Top-k superposition: keep the k_out positions with the highest overlap count.

    Ties are broken uniformly at random.  k_out defaults to the sparsity of the
    first input vector.
    """
    rng = np.random.default_rng() if rng is None else rng
    vecs = list(vecs)
    if not vecs:
        raise ValueError("bundle() needs at least one vector")
    d = len(vecs[0])
    k_out = sparsity(vecs[0]) if k_out is None else int(k_out)
    k_out = min(k_out, d)
    idx = np.concatenate([np.flatnonzero(v) for v in vecs])
    counts = np.bincount(idx, minlength=d).astype(np.float64)
    # Jitter < 1 keeps count order, breaks exact ties uniformly at random.
    scores = counts + rng.random(d)
    if k_out == d:
        chosen = np.arange(d)
    else:
        chosen = np.argpartition(scores, -k_out)[-k_out:]
    out = np.zeros(d, dtype=SPARSE_DTYPE)
    out[chosen] = 1
    return out


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Overlap / k (k = sparsity of a); chance level for unrelated vecs = k/d."""
    k = sparsity(a)
    if k == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / k


def encode_fact(s: np.ndarray, r: np.ndarray, o: np.ndarray,
                perm: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Same convention as src.factworld.encode_fact: bind(bind(S,R), permute(O))."""
    rng = np.random.default_rng() if rng is None else rng
    return bind(bind(s, r, rng=rng), permute(o, perm), rng=rng)


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #
class InvertedIndex:
    """Inverted index position -> object ids, for O(1)-ish sparse cleanup.

    Build once per codebook (CSR layout: sorted positions + owner ids + offsets),
    then ``query`` costs ~k gathers of short posting lists instead of a
    (C x d) matrix product.
    """

    def __init__(self, codebook: dict[str, np.ndarray], d: int | None = None):
        self.names = list(codebook.keys())
        vecs = [np.asarray(codebook[n], dtype=SPARSE_DTYPE) for n in self.names]
        self.d = int(len(vecs[0])) if d is None else int(d)
        self.C = len(vecs)
        pos_parts, owner_parts = [], []
        for i, v in enumerate(vecs):
            p = np.flatnonzero(v)
            pos_parts.append(p.astype(np.int64))
            owner_parts.append(np.full(len(p), i, dtype=np.int64))
        if pos_parts:
            pos = np.concatenate(pos_parts)
            owners = np.concatenate(owner_parts)
        else:  # pragma: no cover - empty codebook
            pos = np.zeros(0, dtype=np.int64)
            owners = np.zeros(0, dtype=np.int64)
        order = np.argsort(pos, kind="stable")
        self._pos = pos[order]
        self._owners = owners[order]
        self._offsets = np.searchsorted(self._pos, np.arange(self.d + 1))
        self._k = sparsity(vecs[0]) if vecs else 0

    def query(self, cand: np.ndarray, k: float | None = None,
              return_sims: bool = False):
        """Return (name, sim) of the best match; sim = overlap / k."""
        k = float(self._k if k is None else k)
        if k <= 0:
            k = 1.0
        counts = np.zeros(self.C, dtype=np.int64)
        for p in np.flatnonzero(cand):
            lo, hi = self._offsets[p], self._offsets[p + 1]
            if hi > lo:
                np.add.at(counts, self._owners[lo:hi], 1)
        i = int(np.argmax(counts))
        sim = float(counts[i]) / k
        if return_sims:
            return self.names[i], sim, counts.astype(np.float64) / k
        return self.names[i], sim


def cleanup(cand: np.ndarray, codebook: dict[str, np.ndarray],
            index: InvertedIndex | None = None,
            k: float | None = None) -> tuple[str, float]:
    """Cleanup of a sparse candidate vector against a codebook."""
    if index is None:
        index = InvertedIndex(codebook)
    return index.query(cand, k=k)

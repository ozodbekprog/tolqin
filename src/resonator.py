"""Resonator networks over FHRR complex vectors (Plate / Frady-style iterative cleanup).

Encoding conventions (documented, consistent across this module and stage 5)
---------------------------------------------------------------------------
Role permutation = ``np.roll`` (src.fhrr.permute).  A *role vector* ``r`` is an
optional extra factor (all-ones means "no role", which is the default for
plain pairs).

Pair::

    pair(a, b)          = bind(permute(a, 1), permute(b, 2))
    pair_r(a, b, r)     = bind(r, pair(a, b))

Triple::

    triple(a, b, c)     = bind(bind(permute(a, 1), permute(b, 2)), permute(c, 3))

Memory is the superposition (bundle) of N encodings.

Decoding
--------
With ``M ~= sum_i pair(a_i, b_i)``::

    a_est = inv_permute(unbind(M, bind(r, permute(b_hat, 2))), 1)
    b_est = inv_permute(unbind(M, bind(r, permute(a_hat, 1))), 2)

i.e. each factor is estimated by unbinding the *other* (current best estimate)
and cleaning up against its codebook.  The inverse permutation is required
because the factor was stored in permuted role position.

``cleanup_fn`` contract: ``cleanup_fn(vec) -> (name, codebook_vector)``.
``make_cleanup_fn`` builds a *hard* (winner-take-all) function from a
``dict[str, np.ndarray]`` using ``src.fhrr.cleanup_matrix``;
``make_soft_cleanup_fn`` builds a *soft* one that returns the softmax-weighted
superposition of the codebook (the classical resonator state).  Iteration stops
when all cleaned-up names are unchanged for one full iteration
(``converged``), or after ``iters`` iterations.

Why the initial state matters (measured, scripts/stage5_factorization.py)
-----------------------------------------------------------------------
Initialise with the *uniform superposition of the codebook* (equal prior over
all symbols).  With that prior both cleanup variants recover isolated pairs
(~100%, ~2 iterations).  With an all-ones initial state the first estimate of
the other factor is zero-information, hard cleanup picks a random symbol and
the network locks into a spurious self-consistent pair (~1-5% accuracy), while
a soft state still converges (~100%): the soft state keeps every candidate
alive and sharpens over iterations, which is the classical resonator dynamic.
Under heavy superposition (bundling many pairs into one memory) neither variant
bootstraps without a cue for at least one factor - the first estimate's
coherence falls below the cleanup noise floor.
"""
from __future__ import annotations

import numpy as np

from src.fhrr import (DTYPE, bind, build_obj_matrix, cleanup_matrix,
                      inv_permute, permute, unbind)


def _ones(d: int) -> np.ndarray:
    return np.ones(d, dtype=DTYPE)


def pair(a: np.ndarray, b: np.ndarray, perm_a: int = 1, perm_b: int = 2) -> np.ndarray:
    return bind(permute(a, perm_a), permute(b, perm_b))


def pair_r(a: np.ndarray, b: np.ndarray, r: np.ndarray,
           perm_a: int = 1, perm_b: int = 2) -> np.ndarray:
    return bind(r, pair(a, b, perm_a=perm_a, perm_b=perm_b))


def triple(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           perm_a: int = 1, perm_b: int = 2, perm_c: int = 3) -> np.ndarray:
    return bind(bind(permute(a, perm_a), permute(b, perm_b)), permute(c, perm_c))


def bundle(vecs) -> np.ndarray:
    """Plain superposition (sum).  Not renormalised; unbinding is unitary."""
    return np.sum(np.stack(list(vecs)), axis=0).astype(DTYPE)


def make_cleanup_fn(codebook: dict[str, np.ndarray]):
    """Build ``cleanup_fn(vec) -> (name, codebook_vector)`` from a codebook."""
    names, mat_n = build_obj_matrix(codebook)
    vecs = [np.asarray(codebook[n], dtype=DTYPE) for n in names]

    def cleanup_fn(vec: np.ndarray):
        name, sim, _ = cleanup_matrix(vec, names, mat_n)
        return name, vecs[names.index(name)]

    cleanup_fn.names = names
    return cleanup_fn


def make_soft_cleanup_fn(codebook: dict[str, np.ndarray], beta: float = 10.0):
    """Soft cleanup: ``(best_name, sum_j softmax(beta*sim_j) * vec_j)``.

    The returned vector is a weighted superposition of the whole codebook (the
    classical resonator state), not a single codebook item.  ``beta`` controls
    how sharply the weights concentrate; 10 works well for d=2048, C=100.
    """
    names, mat_n = build_obj_matrix(codebook)
    vecs = np.stack([np.asarray(codebook[n], dtype=DTYPE) for n in names])

    def cleanup_fn(vec: np.ndarray):
        name, _, sims = cleanup_matrix(vec, names, mat_n)
        w = np.exp(beta * (sims - sims.max()))
        w /= w.sum()
        return name, np.tensordot(w.astype(np.float32), vecs, axes=(0, 0)).astype(DTYPE)

    cleanup_fn.names = names
    return cleanup_fn


def factorize_2(M: np.ndarray, r: np.ndarray | None, a_hat: np.ndarray,
                b_hat: np.ndarray, cleanup_fn, perm_a: int = 1, perm_b: int = 2,
                iters: int = 10, cleanup_b=None) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Iteratively factorise ``M ~= sum bind(r, bind(perm(a,perm_a), perm(b,perm_b)))``.

    ``r=None`` means the all-ones role vector (plain pairs).
    ``cleanup_fn`` cleans the ``a`` factor; ``cleanup_b`` (optional, defaults to
    ``cleanup_fn``) cleans the ``b`` factor, so the two factors may use
    different codebooks.
    Returns ``(a_hat, b_hat, n_iters, converged)`` where ``a_hat``/``b_hat`` are
    codebook/state vectors (unpermuted).
    """
    d = M.shape[0]
    r = _ones(d) if r is None else r
    cleanup_b = cleanup_fn if cleanup_b is None else cleanup_b
    prev = None
    n_iters = 0
    converged = False
    a_name = b_name = None
    for it in range(int(iters)):
        n_iters = it + 1
        a_est = inv_permute(unbind(M, bind(r, permute(b_hat, perm_b))), perm_a)
        a_name, a_hat = cleanup_fn(a_est)
        b_est = inv_permute(unbind(M, bind(r, permute(a_hat, perm_a))), perm_b)
        b_name, b_hat = cleanup_b(b_est)
        if prev == (a_name, b_name):
            converged = True
            break
        prev = (a_name, b_name)
    return a_hat, b_hat, n_iters, converged


def factorize_3(M: np.ndarray, r: np.ndarray | None, a_hat: np.ndarray,
                b_hat: np.ndarray, c_hat: np.ndarray, cleanup_a, cleanup_b,
                cleanup_c, perm_a: int = 1, perm_b: int = 2, perm_c: int = 3,
                iters: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Iteratively factorise a triple superposition ``sum bind(r, triple(a,b,c))``.

    Each factor is estimated by unbinding the product of the other two current
    estimates.  Returns ``(a_hat, b_hat, c_hat, n_iters, converged)``.
    """
    d = M.shape[0]
    r = _ones(d) if r is None else r
    prev = None
    n_iters = 0
    converged = False
    names = (None, None, None)
    for it in range(int(iters)):
        n_iters = it + 1
        a_est = inv_permute(
            unbind(M, bind(r, bind(permute(b_hat, perm_b), permute(c_hat, perm_c)))),
            perm_a)
        a_name, a_hat = cleanup_a(a_est)
        b_est = inv_permute(
            unbind(M, bind(r, bind(permute(a_hat, perm_a), permute(c_hat, perm_c)))),
            perm_b)
        b_name, b_hat = cleanup_b(b_est)
        c_est = inv_permute(
            unbind(M, bind(r, bind(permute(a_hat, perm_a), permute(b_hat, perm_b)))),
            perm_c)
        c_name, c_hat = cleanup_c(c_est)
        if prev == (a_name, b_name, c_name):
            converged = True
            break
        prev = (a_name, b_name, c_name)
    return a_hat, b_hat, c_hat, n_iters, converged

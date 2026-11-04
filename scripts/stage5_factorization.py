"""Stage 5: resonator networks for factorising FHRR pair (and triple) memories.

Encoding (src.resonator, documented there)::

    pair(a, b)   = bind(permute(a, 1), permute(b, 2))
    triple(a,b,c)= bind(bind(permute(a,1), permute(b,2)), permute(c,3))

d=2048, codebooks of 100 symbols for a and b (and c for the triple), FHRR
(unit-magnitude complex) vectors.

CSV: results/stage5_factorization.csv, columns N,d,method,accuracy,mean_iters.

Methods
-------
resonator2              per-pair memory P (N trials).  Init = uniform
                        superposition of each codebook (equal prior over all
                        symbols), soft cleanup, <= 10 iterations.  Correct =
                        both recovered arguments correct.
resonator2_hard         same run with winner-take-all (hard) cleanup; both
                        variants reach ~100% on isolated pairs, i.e. for a
                        single product the prior matters more than softness.
simple_cleanup          non-iterative baseline on the same P:
                        cleanup(unbind(P, permute(b_true, 2))), i.e. b is given
                        (ground truth) and only a is estimated; mean_iters = 1.
                        Isolates the iteration gain: resonator2 must match it
                        *without* knowing b.
resonator2_bundled      one memory bundling all N pairs, each tagged with a
                        unique per-fact address s: M = sum bind(s_i, pair_i).
                        The address is the ``r`` argument of factorize_2 and is
                        the query cue; without it a bundled memory is ambiguous
                        (any stored pair is a valid factorisation).  Soft
                        cleanup, uninformative init.
simple_cleanup_bundled  same bundled M and cue, ground-truth b, one cleanup.
                        This is the capacity reference for the bundle.
resonator3              per-pair triple at N=100 (three unknowns), soft cleanup.
resonator3_cued         triple with c given (isolated a/b problem), soft cleanup.

Note on the bundled numbers: at d=2048, C=100 the uniform prior makes the first
estimate only ~1/sqrt(C + N) coherent (the N-1 other stored pairs and the C-1
wrong codebook symbols both contribute noise, and the prior scales that noise by
sqrt(C)).  Once N grows past ~10 that falls below the cleanup noise floor
sqrt(2 ln C / d) = 0.067 and the network cannot bootstrap; unbinding with the
ground-truth b (simple_cleanup_bundled) stays exact at N=100 and degrades only
with the superposition noise (0.64 at N=500, 0.33 at N=1000).  The per-pair and
cued variants show that the iteration itself is fine (~2-3 iterations, 100%).
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.fhrr import DTYPE, bind, inv_permute, permute, random_fhrr, unbind  # noqa: E402
from src.resonator import (bundle, factorize_2, factorize_3,  # noqa: E402
                           make_cleanup_fn, make_soft_cleanup_fn, pair, triple)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage5_factorization.csv"

D = 2048
C = 100
N_LIST = [100, 500, 1000]
SEED = 0
BETA = 10.0
ITERS = 10
SOFT_DEADLINE_S = 240.0


def _rows_for_N(N: int, rng: np.random.Generator) -> list[dict]:
    A = {f"a{i}": random_fhrr(D, rng) for i in range(C)}
    B = {f"b{i}": random_fhrr(D, rng) for i in range(C)}
    ca_hard = make_cleanup_fn(A)
    cb_hard = make_cleanup_fn(B)
    ca_soft = make_soft_cleanup_fn(A, BETA)
    cb_soft = make_soft_cleanup_fn(B, BETA)
    # uniform prior over symbols: mean of each codebook (all candidates equal)
    a0 = np.mean(list(A.values()), axis=0).astype(DTYPE)
    b0 = np.mean(list(B.values()), axis=0).astype(DTYPE)

    facts = [(f"a{int(rng.integers(C))}", f"b{int(rng.integers(C))}") for _ in range(N)]
    pairs = [pair(A[a], B[b]) for a, b in facts]

    rows = []

    # --- per-pair resonator (soft) -----------------------------------------
    ok, iters = 0, []
    for (an, bn), P in zip(facts, pairs):
        av, bv, ni, _cv = factorize_2(P, None, a0, b0, ca_soft, 1, 2, ITERS,
                                      cleanup_b=cb_soft)
        ok += int(ca_soft(av)[0] == an and cb_soft(bv)[0] == bn)
        iters.append(ni)
    rows.append({"N": N, "d": D, "method": "resonator2",
                 "accuracy": ok / N, "mean_iters": float(np.mean(iters))})

    # --- per-pair resonator (hard cleanup, failed fixpoint mode) ------------
    ok, iters = 0, []
    for (an, bn), P in zip(facts, pairs):
        av, bv, ni, _cv = factorize_2(P, None, a0, b0, ca_hard, 1, 2, ITERS,
                                      cleanup_b=cb_hard)
        ok += int(ca_hard(av)[0] == an and cb_hard(bv)[0] == bn)
        iters.append(ni)
    rows.append({"N": N, "d": D, "method": "resonator2_hard",
                 "accuracy": ok / N, "mean_iters": float(np.mean(iters))})

    # --- non-iterative baseline with ground-truth b -------------------------
    ok = 0
    for (an, bn), P in zip(facts, pairs):
        cand = inv_permute(unbind(P, permute(B[bn], 2)), 1)
        ok += int(ca_hard(cand)[0] == an)
    rows.append({"N": N, "d": D, "method": "simple_cleanup",
                 "accuracy": ok / N, "mean_iters": 1.0})

    # --- bundled memory with a per-fact address cue -------------------------
    S = {f"s{i}": random_fhrr(D, rng) for i in range(N)}
    idx = [f"s{i}" for i in range(N)]
    M = bundle([bind(S[idx[i]], pairs[i]) for i in range(N)])

    ok, iters = 0, []
    for i, (an, bn) in enumerate(facts):
        av, bv, ni, _cv = factorize_2(M, S[idx[i]], a0, b0, ca_soft, 1, 2,
                                      ITERS, cleanup_b=cb_soft)
        ok += int(ca_soft(av)[0] == an and cb_soft(bv)[0] == bn)
        iters.append(ni)
    rows.append({"N": N, "d": D, "method": "resonator2_bundled",
                 "accuracy": ok / N, "mean_iters": float(np.mean(iters))})

    ok = 0
    for i, (an, bn) in enumerate(facts):
        cand = inv_permute(unbind(unbind(M, S[idx[i]]), permute(B[bn], 2)), 1)
        ok += int(ca_hard(cand)[0] == an)
    rows.append({"N": N, "d": D, "method": "simple_cleanup_bundled",
                 "accuracy": ok / N, "mean_iters": 1.0})

    # --- 3-argument extension (N=100 only) ---------------------------------
    if N == 100:
        Cb = {f"c{i}": random_fhrr(D, rng) for i in range(C)}
        cc_hard = make_cleanup_fn(Cb)
        cc_soft = make_soft_cleanup_fn(Cb, BETA)
        facts3 = [(f"a{int(rng.integers(C))}", f"b{int(rng.integers(C))}",
                   f"c{int(rng.integers(C))}") for _ in range(N)]
        c0 = np.mean(list(Cb.values()), axis=0).astype(DTYPE)
        ok, iters = 0, []
        for an, bn, cn in facts3:
            P3 = triple(A[an], B[bn], Cb[cn])
            av, bv, cv, ni, _cv = factorize_3(P3, None, a0, b0, c0,
                                              ca_soft, cb_soft, cc_soft,
                                              iters=ITERS)
            ok += int(ca_soft(av)[0] == an and cb_soft(bv)[0] == bn
                      and cc_soft(cv)[0] == cn)
            iters.append(ni)
        rows.append({"N": N, "d": D, "method": "resonator3",
                     "accuracy": ok / N, "mean_iters": float(np.mean(iters))})

        # c given: unbind the c role, leaving a pair problem (iteration gain)
        ok, iters = 0, []
        for an, bn, cn in facts3:
            P3 = triple(A[an], B[bn], Cb[cn])
            M2 = unbind(P3, permute(Cb[cn], 3))
            av, bv, ni, _cv = factorize_2(M2, None, a0, b0, ca_soft, 1, 2,
                                          ITERS, cleanup_b=cb_soft)
            ok += int(ca_soft(av)[0] == an and cb_soft(bv)[0] == bn)
            iters.append(ni)
        rows.append({"N": N, "d": D, "method": "resonator3_cued",
                     "accuracy": ok / N, "mean_iters": float(np.mean(iters))})

    return rows


def main() -> None:
    t_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    all_rows = []
    for N in N_LIST:
        if time.perf_counter() - t_start > SOFT_DEADLINE_S:
            print(f"N={N}: skipped (soft deadline {SOFT_DEADLINE_S:.0f}s)")
            continue
        t0 = time.perf_counter()
        rows = _rows_for_N(N, rng)
        all_rows.extend(rows)
        for r in rows:
            print(f"N={r['N']:>5}  {r['method']:<22} acc={r['accuracy']:.3f}  "
                  f"iters={r['mean_iters']:.2f}")
        print(f"  ({time.perf_counter() - t0:.1f}s)\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "d", "method", "accuracy", "mean_iters"])
        w.writeheader()
        for r in all_rows:
            w.writerow({"N": r["N"], "d": r["d"], "method": r["method"],
                        "accuracy": f"{r['accuracy']:.4f}",
                        "mean_iters": f"{r['mean_iters']:.2f}"})
    print(f"wrote {OUT}")
    print(f"total wall clock: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

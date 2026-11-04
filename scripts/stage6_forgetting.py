"""Stage 6 — FSRS forgetting + LP attention on top of one FHRR bundle.

World: d=1024, N=1000 facts, seed=3. Simulate 2000 steps, dt=1. At each step
with probability 0.3 pick a random stored fact, query it, and review it through
FSRS (success = cleanup returned the true object). Every 50 steps old facts
(R < 0.1) are pruned; every 100 steps the state is measured with 50 read-only
probes (probes do NOT review).

CSV 1 — results/stage6_forgetting_curve.csv
    step            simulated step (1..2000)
    accuracy        share of probe facts whose cleanup name == true object
    mean_R          mean FSRS retention over stored facts
    fraction_alive  share of stored facts with R > 0.5
    n_pruned        cumulative number of facts pruned (R < 0.1)

CSV 2 — results/stage6_weighted_vs_flat.csv (same query schedule / seed)
    method          "weighted" (w_i = R_i) or "flat" (w_i = 1)
    accuracy        share of scheduled main queries answered correctly
    steps           number of simulated steps

Both variants draw review decisions and fact picks from one identically seeded
RNG, so the schedule matches; the trajectories diverge only through each
variant's own retrieval outcomes.

Note on the first ~800 steps: d=1024 with 1000 stored facts is past the FHRR
cleanup capacity (a flat bundle answers ~3-4% of 1000 facts correctly), so
probe accuracy sits near the floor until pruning removes enough clutter. That
is the measured behaviour, not a bug.

Timing constants: S0 = 10 gives a half-life of 90 steps (R = 0.1 after 810
steps). With the specified review rate each fact is reviewed only ~0.6 times
in 2000 steps, so decay dominates and the module default S0 = 1 would prune
the entire memory within ~100 steps. S0 = 10 keeps the whole decay -> prune ->
recovery arc inside the horizon. Bundles are rebalanced every 50 queries (a
full rebuild per query would be ~1M complex MACs each time).
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.factworld import encode_fact, make_world          # noqa: E402
from src.integrated_memory import IntegratedMemory        # noqa: E402

D = 1024
N_FACTS = 1000
WORLD_SEED = 3
STEPS = 2000
REVIEW_P = 0.3
S0 = 10.0
PRUNE_THRESHOLD = 0.1
PRUNE_EVERY = 50
REBALANCE_EVERY = 50
MEASURE_EVERY = 100
PROBE_N = 50
SCHEDULE_SEED = 2026
PROBE_SEED = 777
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def simulate(fw: dict, vectors: np.ndarray, weighted: bool, schedule_seed: int):
    """Run one simulation. Returns (curve rows, main-query accuracy, n_pruned)."""
    im = IntegratedMemory(D, perm=1, world=fw, S0=S0,
                          rebalance_every=REBALANCE_EVERY, weighted=weighted)
    for (s, r, o), fvec in zip(fw["facts"], vectors):
        im.add_fact(s, r, o, t=0.0, fvec=fvec)
    im.rebalance(0.0)

    rng = np.random.default_rng(schedule_seed)
    rng_probe = np.random.default_rng(PROBE_SEED)
    rows = []
    sched_ok = 0
    sched_n = 0

    for t in range(1, STEPS + 1):
        # --- one review event ------------------------------------------------
        if im.facts and rng.random() < REVIEW_P:
            fact = im.facts[int(rng.integers(len(im.facts)))]
            name, _ = im.query(fact["subject"], fact["relation"], t)
            ok = name == fact["object"]
            im.recall(fact["id"], t, ok)
            sched_ok += int(ok)
            sched_n += 1

        # --- forgetting / pruning -------------------------------------------
        if t % PRUNE_EVERY == 0:
            im.prune(t, PRUNE_THRESHOLD)

        # --- measurement ------------------------------------------------------
        if t % MEASURE_EVERY == 0:
            n = min(PROBE_N, len(im.facts))
            if n:
                idx = rng_probe.choice(len(im.facts), size=n, replace=False)
                correct = 0
                for j in idx:
                    f = im.facts[j]
                    pred, _ = im.query(f["subject"], f["relation"], t)
                    correct += int(pred == f["object"])
                acc = correct / n
            else:
                acc = float("nan")               # empty memory: nothing to probe
            rows.append({
                "step": t,
                "accuracy": acc,
                "mean_R": im.mean_R(t),
                "fraction_alive": im.fraction_retained(t, 0.5),
                "n_pruned": im.n_pruned,
            })

    sched_acc = sched_ok / sched_n if sched_n else float("nan")
    return rows, sched_acc, sched_n, im.n_pruned, im.n_facts, im


def main() -> None:
    t_start = time.time()
    fw = make_world(n_facts=N_FACTS, d=D, seed=WORLD_SEED)
    vectors = np.stack([encode_fact(fw, s, r, o) for s, r, o in fw["facts"]])

    rows, sched_acc, n_q, n_pruned, n_left, im = simulate(fw, vectors, True, SCHEDULE_SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    curve_path = RESULTS / "stage6_forgetting_curve.csv"
    with curve_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step", "accuracy", "mean_R",
                                           "fraction_alive", "n_pruned"])
        w.writeheader()
        for row in rows:
            w.writerow({"step": row["step"],
                        "accuracy": f"{row['accuracy']:.4f}",
                        "mean_R": f"{row['mean_R']:.4f}",
                        "fraction_alive": f"{row['fraction_alive']:.4f}",
                        "n_pruned": row["n_pruned"]})

    # --- weighted vs flat on the same query schedule ------------------------
    _, flat_acc, flat_q, flat_pruned, flat_left, _ = simulate(
        fw, vectors, False, SCHEDULE_SEED)
    cmp_path = RESULTS / "stage6_weighted_vs_flat.csv"
    with cmp_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "accuracy", "steps"])
        w.writerow(["weighted", f"{sched_acc:.4f}", STEPS])
        w.writerow(["flat", f"{flat_acc:.4f}", STEPS])

    elapsed = time.time() - t_start
    first, last = rows[0], rows[-1]
    print(f"world d={D} N={N_FACTS} seed={WORLD_SEED}  steps={STEPS}  S0={S0}")
    print(f"{'step':>6} {'accuracy':>9} {'mean_R':>7} {'alive':>6} {'pruned':>7}")
    for row in (rows[0], rows[len(rows) // 4], rows[len(rows) // 2],
                rows[3 * len(rows) // 4], rows[-1]):
        print(f"{row['step']:>6} {row['accuracy']:>9.3f} {row['mean_R']:>7.3f} "
              f"{row['fraction_alive']:>6.3f} {row['n_pruned']:>7d}")
    print(f"curve endpoints: accuracy {first['accuracy']:.3f} -> {last['accuracy']:.3f}, "
          f"mean_R {first['mean_R']:.3f} -> {last['mean_R']:.3f}, "
          f"alive {first['fraction_alive']:.3f} -> {last['fraction_alive']:.3f}, "
          f"pruned {last['n_pruned']}")
    print(f"main queries: {n_q} (weighted), {flat_q} (flat); "
          f"stored at end: {n_left} weighted / {flat_left} flat")
    print(f"weighted acc {sched_acc:.3f} vs flat acc {flat_acc:.3f}")
    print(f"pruned: {n_pruned} weighted / {flat_pruned} flat")
    att = im.attn.weights(float(STEPS), dt=1.0)
    if att:
        top = sorted(att.values(), reverse=True)[:max(1, len(att) // 10)]
        print(f"LP attention at t={STEPS}: {len(att)} tracked facts, "
              f"top-10% share {sum(top):.3f} (uniform {0.1:.3f})")
    print(f"runtime {elapsed:.1f} s -> {curve_path.name}, {cmp_path.name}")


if __name__ == "__main__":
    main()

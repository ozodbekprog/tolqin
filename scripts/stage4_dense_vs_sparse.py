"""Stage 4: dense FHRR vs sparse binary VSA (SBC) on the same fact-retrieval task.

Both methods store the same facts ``(s, r, o)`` with the same encoding
convention ``fact = bind(bind(S, R), permute(O, 1))`` and answer queries
``(s, r) -> o`` by unbinding the query key and cleaning up against the object
codebook.  Objects are unique per fact.

Dense side
    src.fhrr + src.factworld + src.blocked_memory, d=2048, B=8 blocks,
    strategy="round_robin", perm=1.
Sparse side
    src.sparse_vsa, d=10000, k=100, no blocks (single bundled memory).
Facts
    Facts are assigned one-to-one to unique (s, r) keys drawn from a
    ceil(sqrt(N)) x ceil(sqrt(N)) subject/relation grid built by factworld.
    (make_world's random assignment would alias many facts onto the same
    (s, r) at large N, making the query itself ambiguous; the codebooks still
    come from src.factworld.)  Encoding: ``fact = bind(bind(S, R), permute(O, 1))``.
Sparse caveat
    Sparse ``bind`` is randomised (thinning), so the derived query keys
    ``bind(S, R)`` are computed once and reused at encode and query time;
    recomputing them per query would break the (already approximate)
    unbinding.  Cleanup uses an inverted index over the object codebook.

CSV: results/stage4_dense_vs_sparse.csv with columns
    method      - "dense" | "sparse"
    d           - dimensionality
    k           - ones per vector (sparse); 0 for dense
    B           - blocks (dense); 1 = one bundled memory (sparse)
    N           - stored facts
    accuracy    - fraction of queries whose cleaned-up object equals the truth
    ms_per_query- mean wall-clock per query (unbind + cleanup), milliseconds
    memory_MB   - (object codebook + memory vector/blocks) bytes / 1e6

Queries: all N facts for N <= 1000, otherwise 200 uniformly sampled facts.

Timing is query-only; encoding/bundling/index building are excluded (the same
applies to both methods).  A soft deadline keeps the whole script under five
minutes; sparse configurations that would exceed it are skipped and noted.

What the numbers mean
---------------------
Dense: within one block the retrieved object's similarity is ~1/sqrt(N/B), the
cleanup noise floor over N unique objects is ~sqrt(2 ln N / d).  At d=2048 and
N=1000 (125 facts per block) these are comparable, so accuracy holds (~0.96);
at N>=5000 (625+ facts per block) the signal drops below the noise floor and
dense retrieval collapses even though there are 8 blocks.  Blocks reduce
interference but every block is cleaned against the same N-object codebook.
Sparse: a single top-k bundle keeps only k=100 active bits, i.e. ~log2 C(10000,100)
bits of state no matter how many facts were added, and thinning-unbinding puts
the object signal mostly in the random filler positions.  The result is a
working memory for 1-2 facts only (see the bracket printed at the end); the
8x smaller uint8 codebook does not compensate.  This is a property of the
thinning bind + top-k bundle scheme, not of the index.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.blocked_memory import BlockedMemory  # noqa: E402
from src.factworld import encode_fact, make_world, query_key  # noqa: E402
from src.fhrr import build_obj_matrix  # noqa: E402
from src.sparse_vsa import (DEFAULT_D, DEFAULT_K, InvertedIndex, bind,  # noqa: E402
                            bundle, inv_permute, permute, random_sparse, unbind)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage4_dense_vs_sparse.csv"

D_DENSE, B_DENSE = 2048, 8
N_LIST = [1000, 5000, 10000]
N_QUERIES_BIG = 200
SEED = 0
SOFT_DEADLINE_S = 270.0


def make_unique_fact_world(N: int, d: int, seed: int = 0) -> dict:
    """factworld codebooks, but facts on unique (s, r) keys (one fact per key)."""
    K = int(np.ceil(np.sqrt(N)))
    fw = make_world(n_facts=N, d=d, n_subjects=K, n_relations=K, seed=seed)
    keys = [(f"s{i}", f"r{j}") for i in range(K) for j in range(K)]
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(keys))[:N]
    fw["facts"] = [(keys[k][0], keys[k][1], f"o{i}") for i, k in enumerate(order)]
    return fw


def dense_run(N: int, rng: np.random.Generator) -> dict:
    fw = make_unique_fact_world(N, D_DENSE, seed=SEED)
    bm = BlockedMemory(D_DENSE, n_blocks=B_DENSE, strategy="round_robin", perm=1,
                       seed=SEED)
    for s, r, o in fw["facts"]:
        bm.add(encode_fact(fw, s, r, o), subject=s, relation=r)
    names, mat_n = build_obj_matrix(fw["objects"])

    qidx = np.arange(N) if N <= 1000 else rng.choice(N, N_QUERIES_BIG, replace=False)
    correct = 0
    t0 = time.perf_counter()
    for i in qidx:
        s, r, o = fw["facts"][i]
        name, _sim, _nb = bm.query(query_key(fw, s, r), names, mat_n, perm_used=1)
        correct += int(name == o)
    elapsed = time.perf_counter() - t0
    mem_bytes = bm.memory_bytes() + sum(v.nbytes for v in fw["objects"].values())
    return {"method": "dense", "d": D_DENSE, "k": 0, "B": B_DENSE, "N": N,
            "accuracy": correct / len(qidx),
            "ms_per_query": 1000.0 * elapsed / len(qidx),
            "memory_MB": mem_bytes / 1e6}


def sparse_setup(N: int, rng: np.random.Generator, d: int = DEFAULT_D,
                 k: int = DEFAULT_K):
    K = int(np.ceil(np.sqrt(N)))
    subs = {f"s{i}": random_sparse(d, k, rng) for i in range(K)}
    rels = {f"r{i}": random_sparse(d, k, rng) for i in range(K)}
    objs = {f"o{i}": random_sparse(d, k, rng) for i in range(N)}
    keys = {(s, r): bind(subs[s], rels[r], rng) for s in subs for r in rels}
    grid = [(f"s{i}", f"r{j}") for i in range(K) for j in range(K)]
    order = rng.permutation(len(grid))[:N]
    facts = [(grid[j][0], grid[j][1], f"o{i}") for i, j in enumerate(order)]
    # fact = bind(bind(S, R), permute(O, 1)) with the cached key bind(S, R)
    mem = bundle([bind(keys[(s, r)], permute(objs[o], 1), rng)
                  for s, r, o in facts], k_out=k, rng=rng)
    index = InvertedIndex(objs, d=d)
    return facts, keys, objs, mem, index


def sparse_run(N: int, rng: np.random.Generator) -> dict:
    facts, keys, objs, mem, index = sparse_setup(N, rng)
    qidx = np.arange(N) if N <= 1000 else rng.choice(N, N_QUERIES_BIG, replace=False)
    correct = 0
    t0 = time.perf_counter()
    for i in qidx:
        s, r, o = facts[i]
        cand = inv_permute(unbind(mem, keys[(s, r)], rng), 1)
        name, _sim = index.query(cand, k=DEFAULT_K)
        correct += int(name == o)
    elapsed = time.perf_counter() - t0
    mem_bytes = mem.nbytes + sum(v.nbytes for v in objs.values())
    return {"method": "sparse", "d": DEFAULT_D, "k": DEFAULT_K, "B": 1, "N": N,
            "accuracy": correct / len(qidx),
            "ms_per_query": 1000.0 * elapsed / len(qidx),
            "memory_MB": mem_bytes / 1e6}


def sparse_capacity_bracket(rng: np.random.Generator):
    """Diagnostic: single-memory sparse capacity (why N >= 1000 is at chance)."""
    print("\nsparse capacity bracket (same pipeline, single 100-bit memory):")
    for n in (1, 2, 5, 10, 50):
        r = sparse_run(n, rng)
        print(f"  N={n:>5}  accuracy={r['accuracy']:.3f}  "
              f"{r['ms_per_query']:.3f} ms/query")
    print("  chance level for N objects = 1/N; N=1..2 is the working regime.")


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    t_start = time.perf_counter()
    for N in N_LIST:
        t0 = time.perf_counter()
        r = dense_run(N, rng)
        rows.append(r)
        print(f"dense  N={N:>6}  acc={r['accuracy']:.3f}  "
              f"{r['ms_per_query']:.2f} ms/query  {r['memory_MB']:.1f} MB  "
              f"({time.perf_counter() - t0:.1f}s)")
    for N in N_LIST:
        if time.perf_counter() - t_start > SOFT_DEADLINE_S:
            print(f"sparse N={N}: skipped (soft deadline {SOFT_DEADLINE_S:.0f}s)")
            continue
        t0 = time.perf_counter()
        r = sparse_run(N, rng)
        rows.append(r)
        print(f"sparse N={N:>6}  acc={r['accuracy']:.3f}  "
              f"{r['ms_per_query']:.2f} ms/query  {r['memory_MB']:.1f} MB  "
              f"({time.perf_counter() - t0:.1f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "d", "k", "B", "N", "accuracy", "ms_per_query", "memory_MB"]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**r,
                        "accuracy": f"{r['accuracy']:.4f}",
                        "ms_per_query": f"{r['ms_per_query']:.3f}",
                        "memory_MB": f"{r['memory_MB']:.3f}"})
    print(f"\nwrote {OUT}")

    sparse_capacity_bracket(rng)
    print(f"total wall clock: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()

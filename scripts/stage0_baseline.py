"""Stage 0 — klassik real HRR (circular convolution) baseline o'lchovlari.

Ishga tushirish (repo ildizidan):
    python3 scripts/stage0_baseline.py

CSV ustunlari (results/baseline_results.csv):
    d            : vektor o'lchami
    B            : bloklar soni (bu baseline'da doim 1)
    N            : faktlar soni (= kodkitob hajmi C; har faktga unikal obyekt)
    cleanup      : "proper" yoki "elementwise_buggy"
    learn_ms     : barcha faktni kodlash + bitta M ga bundle qilish vaqti (ms)
    query_ms     : barcha N so'rov uchun jami vaqt (ms, retrieve + cleanup)
    accuracy     : to'g'ri topilgan faktlar ulushi [0,1]
    mem_MB       : M (float64) + obyekt kodkitobi (float64) baytlari, MB
    predicted_acc: reja bashorati (faqat d=512, N=1000 uchun 0.20), aks holda bo'sh
    delta        : accuracy - predicted_acc (bashorat yo'q bo'lsa bo'sh)

Eslatma: predicted_acc 0.20 — reja "eski kod d=512, N=1000 da 15-25% beradi"
degan bashorati; u to'g'ri (proper) cleanup yo'liga tegishli.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.baseline_hrr import (
    accuracy_matrix,
    build_obj_matrix,
    bundle,
    cleanup_buggy_matrix,
    cleanup_proper_matrix,
    encode_fact,
    make_world,
    quantize_elementwise,
)

D_LIST = (256, 512, 1024)
N_LIST = (200, 1000)
N_SUBJECTS = 50
N_RELATIONS = 20
PREDICTED = {(512, 1000): 0.20}
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "baseline_results.csv"


def run_config(d: int, N: int, reps: int = 3) -> list[dict]:
    fw = make_world(n_facts=N, d=d, n_subjects=N_SUBJECTS,
                    n_relations=N_RELATIONS, seed=0)

    # Vaqtlar shovqinli muhitda o'lchanadi -> reps marta o'lchab min olinadi
    learn_ms = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        M = bundle(encode_fact(fw, s, r, o) for s, r, o in fw["facts"])
        learn_ms = min(learn_ms, (time.perf_counter() - t0) * 1e3)

    mem_bytes = M.nbytes + sum(v.nbytes for v in fw["objects"].values())
    mem_MB = mem_bytes / (1024 * 1024)

    names, mat_n = build_obj_matrix(fw["objects"])
    mat_q = quantize_elementwise(mat_n)   # xato varianti uchun bir marta

    rows = []
    for cleanup_name, fn in (
        ("proper", cleanup_proper_matrix),
        ("elementwise_buggy", lambda r, n, m: cleanup_buggy_matrix(r, n, m, mat_q=mat_q)),
    ):
        acc, query_ms = 0.0, float("inf")
        for _ in range(reps):
            a, ms = accuracy_matrix(fw, M, names, mat_n, fn)
            acc, query_ms = a, min(query_ms, ms)
        pred = PREDICTED.get((d, N))
        rows.append({
            "d": d, "B": 1, "N": N, "cleanup": cleanup_name,
            "learn_ms": round(learn_ms, 3),
            "query_ms": round(query_ms, 3),
            "accuracy": round(acc, 4),
            "mem_MB": round(mem_MB, 4),
            "predicted_acc": "" if pred is None else f"{pred:.2f}",
            "delta": "" if pred is None else round(acc - pred, 4),
        })
    return rows


def main() -> None:
    header = ["d", "B", "N", "cleanup", "learn_ms", "query_ms", "accuracy",
              "mem_MB", "predicted_acc", "delta"]
    t_start = time.perf_counter()
    rows: list[dict] = []
    for d in D_LIST:
        for N in N_LIST:
            rows.extend(run_config(d, N))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"Stage 0 — classic real HRR (circular convolution), B=1, "
          f"C=N (unique object per fact), {N_SUBJECTS} subjects x {N_RELATIONS} relations\n")
    print(f"{'d':>5} {'N':>5} {'cleanup':<18} {'learn_ms':>9} {'query_ms':>9} "
          f"{'acc':>6} {'mem_MB':>8} {'pred':>5} {'delta':>7}")
    print("-" * 82)
    for r in rows:
        pred = r["predicted_acc"] if r["predicted_acc"] != "" else "-"
        delta = f"{r['delta']:+.3f}" if r["delta"] != "" else "-"
        print(f"{r['d']:>5} {r['N']:>5} {r['cleanup']:<18} {r['learn_ms']:>9.2f} "
              f"{r['query_ms']:>9.2f} {r['accuracy']:>6.3f} {r['mem_MB']:>8.4f} "
              f"{pred:>5} {delta:>7}")
    print("-" * 82)
    # Kontekst: 50x20 = 1000 kalit maydoni; N fakt tasodifiy taqsimlanadi.
    # Bitta (s,r) ga tushgan faktlar uchun so'rov o'zi aniq emas -> shift.
    for N in N_LIST:
        parts = []
        for d in D_LIST:
            facts = make_world(n_facts=N, d=d, n_subjects=N_SUBJECTS,
                               n_relations=N_RELATIONS, seed=0)["facts"]
            counts = Counter((s, r) for s, r, _ in facts)
            parts.append(f"d={d}:{sum(1.0 / counts[(s, r)] for s, r, _ in facts) / N:.3f}")
        print(f"note: N={N} perfect-retrieval ceiling "
              f"(duplicate (s,r) keys cap accuracy): {', '.join(parts)}")
    print(f"total wall time: {time.perf_counter() - t_start:.2f}s   -> {OUT}")


if __name__ == "__main__":
    main()

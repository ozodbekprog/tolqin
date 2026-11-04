"""Stage 7 — final aggregation: benchmark table, ablations, scaling curves.

Run from the repo root::

    python3 scripts/stage7_benchmark.py

Reads the measured stage CSV files only (no re-simulation, no invented
numbers) and writes:

  results/final_benchmark.csv  task,method,config,accuracy,ms_per_query,memory_MB,source
  results/ablation.csv         component,gain,note
  results/scaling_curves.png   2 panels: accuracy vs B, accuracy vs N (log x)

Column conventions
------------------
accuracy     : share of queries whose cleanup returned the true object [0,1]
ms_per_query : mean query latency in ms (baseline query_ms is a total over N
               queries and is divided by N here)
memory_MB    : resident VSA memory reported by the source stage (codebook +
               blocks); "n/a" where the stage did not measure it
"""
from __future__ import annotations

import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

OUT_BENCH = RESULTS / "final_benchmark.csv"
OUT_ABLATION = RESULTS / "ablation.csv"
OUT_PNG = RESULTS / "scaling_curves.png"

NA = "n/a"


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find(rows: list[dict], **conds) -> dict:
    """First row matching all string/number conditions (numbers compared as float)."""
    for row in rows:
        ok = True
        for key, want in conds.items():
            got = row[key]
            if isinstance(want, (int, float)):
                if to_float(got) != float(want):
                    ok = False
                    break
            elif got != want:
                ok = False
                break
        if ok:
            return row
    raise KeyError(f"no row matching {conds}")


def fmt(value: float | None, digits: int = 4) -> str:
    return NA if value is None else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ------------------------------------------------------------------ load data
baseline = read_csv("baseline_results.csv")
s2 = read_csv("stage2_scaling.csv")
s3 = read_csv("stage3_d_sweep.csv")
cap = read_csv("stage3_capacity_curve.csv")
s4 = read_csv("stage4_dense_vs_sparse.csv")
s5 = read_csv("stage5_factorization.csv")
s6 = read_csv("stage6_forgetting_curve.csv")
s6w = read_csv("stage6_weighted_vs_flat.csv")


# ------------------------------------------------------- final benchmark rows
def bm_row(task, method, config, accuracy, ms_per_query, memory_MB, source):
    return {
        "task": task,
        "method": method,
        "config": config,
        "accuracy": accuracy,
        "ms_per_query": ms_per_query,
        "memory_MB": memory_MB,
        "source": source,
    }


base_d1024_n1000 = find(baseline, d=1024, B=1, N=1000, cleanup="proper")
n1000 = float(base_d1024_n1000["N"])
base_ms = float(base_d1024_n1000["query_ms"]) / n1000  # total -> per query

s3_b1 = find(s3, d=1024, B=1, strategy="round_robin")
s3_b16 = find(s3, d=1024, B=16, strategy="round_robin")
s3_b32 = find(s3, d=1024, B=32, strategy="round_robin")
dense = find(s4, method="dense", d=2048, B=8, N=1000)
sparse = find(s4, method="sparse", d=10000, k=100, N=1000)
weighted = find(s6w, method="weighted")
flat = find(s6w, method="flat")
res_n1000 = find(s5, N=1000, d=2048, method="resonator2")
simple_n1000 = find(s5, N=1000, d=2048, method="simple_cleanup")

bench = [
    bm_row("store_retrieve", "HRR (baseline)", "d=1024,N=1000,B=1,proper",
           base_d1024_n1000["accuracy"], fmt(base_ms, 5),
           base_d1024_n1000["mem_MB"], "baseline_results.csv"),
    bm_row("store_retrieve", "FHRR", "d=1024,B=1",
           s3_b1["accuracy"], s3_b1["ms_per_query"], s3_b1["memory_MB"],
           "stage3_d_sweep.csv"),
    bm_row("store_retrieve", "FHRR", "d=1024,B=16",
           s3_b16["accuracy"], s3_b16["ms_per_query"], s3_b16["memory_MB"],
           "stage3_d_sweep.csv"),
    bm_row("store_retrieve", "FHRR", "d=1024,B=32",
           s3_b32["accuracy"], s3_b32["ms_per_query"], s3_b32["memory_MB"],
           "stage3_d_sweep.csv"),
    bm_row("store_retrieve", "dense", "d=2048,B=8,N=1000",
           dense["accuracy"], dense["ms_per_query"], dense["memory_MB"],
           "stage4_dense_vs_sparse.csv"),
    bm_row("store_retrieve", "sparse", "d=10000,k=100,N=1000",
           sparse["accuracy"], sparse["ms_per_query"], sparse["memory_MB"],
           "stage4_dense_vs_sparse.csv"),
    bm_row("forgetting", "weighted (FSRS)", "step=2000",
           weighted["accuracy"], NA, NA, "stage6_weighted_vs_flat.csv"),
    bm_row("forgetting", "flat", "step=2000",
           flat["accuracy"], NA, NA, "stage6_weighted_vs_flat.csv"),
    bm_row("factorization", "resonator2", "d=2048,N=1000",
           res_n1000["accuracy"], NA, NA, "stage5_factorization.csv"),
    bm_row("factorization", "simple_cleanup", "d=2048,N=1000",
           simple_n1000["accuracy"], NA, NA, "stage5_factorization.csv"),
]

with open(OUT_BENCH, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["task", "method", "config", "accuracy", "ms_per_query",
                       "memory_MB", "source"])
    writer.writeheader()
    writer.writerows(bench)


# --------------------------------------------------------------- ablations
# unique_keys fix: aliased world ceiling 0.631 (docs/capacity_formula.md,
# N=1000) vs collision-free ceiling 1.000 (stage3_d_sweep.csv `ceiling`).
s2_b1 = find(s2, d=512, B=1, strategy="round_robin", query_mode="all")
s2_b8 = find(s2, d=512, B=8, strategy="round_robin", query_mode="all")
s2_b32 = find(s2, d=512, B=32, strategy="round_robin", query_mode="all")
base_proper = find(baseline, d=1024, B=1, N=200, cleanup="proper")
base_buggy = find(baseline, d=1024, B=1, N=200, cleanup="elementwise_buggy")
bundled_res = find(s5, N=1000, d=2048, method="resonator2_bundled")
bundled_simple = find(s5, N=1000, d=2048, method="simple_cleanup_bundled")

unique_gain = float(s3_b1["ceiling"]) - 0.631
ablations = [
    ("unique_keys fix", unique_gain,
     "collision-free vs aliased world: ceiling 0.631 -> 1.000 (N=1000)"),
    ("block scaling B=1->B=8 (d=512)",
     float(s2_b8["accuracy"]) - float(s2_b1["accuracy"]),
     f"stage2 round_robin: {s2_b1['accuracy']} -> {s2_b8['accuracy']}"),
    ("block scaling B=8->B=32 (d=512)",
     float(s2_b32["accuracy"]) - float(s2_b8["accuracy"]),
     f"stage2 round_robin: {s2_b8['accuracy']} -> {s2_b32['accuracy']}"),
    ("proper vs elementwise_buggy cleanup (d=1024,N=200)",
     float(base_proper["accuracy"]) - float(base_buggy["accuracy"]),
     f"baseline: proper {base_proper['accuracy']} vs buggy {base_buggy['accuracy']}"),
    ("resonator2 vs simple cleanup (d=2048,N=1000)",
     float(res_n1000["accuracy"]) - float(simple_n1000["accuracy"]),
     "tie: both 1.000; resonator uses 2.0 iters, simple 1.0 (b given)"),
    ("resonator2_bundled vs simple_cleanup_bundled (N=1000)",
     float(bundled_res["accuracy"]) - float(bundled_simple["accuracy"]),
     f"bundled memory: {bundled_res['accuracy']} vs {bundled_simple['accuracy']}"),
    ("FSRS weighting vs flat (step 2000)",
     float(weighted["accuracy"]) - float(flat["accuracy"]),
     f"weighted {weighted['accuracy']} vs flat {flat['accuracy']} (weighting hurts)"),
]

with open(OUT_ABLATION, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["component", "gain", "note"])
    for component, gain, note in ablations:
        writer.writerow([component, f"{gain:+.4f}", note])


# ---------------------------------------------------------- scaling figure
def unique_s2(d: int) -> tuple[list[int], list[float]]:
    """round_robin `all` rows for one d, deduplicated by B, sorted."""
    points: dict[int, float] = {}
    for row in s2:
        if (int(row["d"]) == d and row["strategy"] == "round_robin"
                and row["query_mode"] == "all"):
            points.setdefault(int(row["B"]), float(row["accuracy"]))
    bs = sorted(points)
    return bs, [points[b] for b in bs]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)

for d, colour, marker in ((512, "#1f77b4", "o"), (1024, "#d62728", "s")):
    bs, accs = unique_s2(d)
    ax1.plot(bs, accs, marker=marker, color=colour, label=f"d={d}")
ax1.set_xscale("log", base=2)
ax1.set_xticks([1, 2, 4, 8, 16, 32])
ax1.set_xticklabels(["1", "2", "4", "8", "16", "32"])
ax1.xaxis.set_minor_formatter(NullFormatter())
ax1.set_xlabel("Blocks B")
ax1.set_ylabel("Accuracy @ N=1000")
ax1.set_title("Block scaling (stage 2, round_robin)")
ax1.set_ylim(-0.02, 1.05)
ax1.grid(alpha=0.3)
ax1.legend()

for d, colour, marker in ((512, "#1f77b4", "o"), (1024, "#d62728", "s"),
                          (2048, "#2ca02c", "^"), (4096, "#9467bd", "D")):
    rows = sorted((r for r in cap if int(r["d"]) == d),
                  key=lambda r: int(r["N"]))
    ns = [int(r["N"]) for r in rows]
    accs = [float(r["accuracy"]) for r in rows]
    ax2.plot(ns, accs, marker=marker, color=colour, label=f"d={d}")
ax2.set_xscale("log")
ax2.set_xticks([50, 100, 200, 400, 800])
ax2.set_xticklabels(["50", "100", "200", "400", "800"])
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xlabel("Facts N (log scale)")
ax2.set_ylabel("Accuracy (B=1)")
ax2.set_title("Capacity curves (stage 3, B=1)")
ax2.set_ylim(-0.02, 1.05)
ax2.grid(alpha=0.3)
ax2.legend()

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)


# ------------------------------------------------------------------- report
print("FINAL BENCHMARK (results/final_benchmark.csv)")
print(f"{'task':<15} {'method':<18} {'config':<24} {'acc':>7} "
      f"{'ms/q':>9} {'MB':>9}  source")
for row in bench:
    print(f"{row['task']:<15} {row['method']:<18} {row['config']:<24} "
          f"{row['accuracy']:>7} {row['ms_per_query']:>9} "
          f"{row['memory_MB']:>9}  {row['source']}")

print("\nHeadline: d=1024/B=32 -> "
      f"{pct(float(s3_b32['accuracy']))} @ N=1000; d=2048/B=8 -> "
      f"{pct(float(dense['accuracy']))}; sparse -> "
      f"{pct(float(sparse['accuracy']))}.")

print("\nABLATIONS (results/ablation.csv)")
for component, gain, note in ablations:
    print(f"  {gain:+.4f}  {component:<52} {note}")

print(f"\nFigure: {OUT_PNG.relative_to(ROOT)}")
print(f"Rows: benchmark={len(bench)}, ablation={len(ablations)}")

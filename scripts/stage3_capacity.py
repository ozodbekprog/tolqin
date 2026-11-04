"""Stage 3 — sig'im injeneriyasi: d x B sweep (N=1000) + sig'im egri chizig'i.

Ishga tushirish:
    cd /Users/probox/Documents/tolqin && python3 scripts/stage3_capacity.py

Natija fayllari va ustunlar
---------------------------
results/stage3_d_sweep.csv
  d             vektor o'lchami (complex64)
  B             bloklar soni
  strategy      "round_robin" | "similarity"
  n_queries     baholangan so'rovlar soni (deterministik, tekis tanlangan indekslar)
  accuracy      to'g'ri cleanup topilgan faktlar ulushi [0, 1]
  ms_per_query  bitta so'rovga o'rtacha vaqt (ms); faqat query, add hisobga olinmaydi
  memory_MB     objects kodkitobi + bloklar hajmi (MB = 1e6 bayt)
  load_ratio    max(counts) / mean(counts); 1.0 = mukammal balans
  ci95          binomial 95% ishonch oralig'i yarim kengligi (accuracy uchun)
  ceiling       shu so'rov to'plamida (s,r) kalit to'qnashuvi bergan kutilayotgan
                accuracy shifti (sum 1/k; nazorat ustuni, natija emas)

results/stage3_capacity_curve.csv
  d, B, N, accuracy     (B=1; accuracy N ta fakt bo'yicha to'liq baholanadi)
                        d=2048 — talab qilingan asosiy egri chiziq; qolgan d lar
                        formulani har bir d uchun solishtirish uchun qo'shilgan

results/stage3_optimal_config.json
  {"d", "B", "strategy", "acc", "ms_per_query", "memory_MB",
   "target_met", "best_acc", ...}

Runtime guard: d >= 2048 yoki B >= 16 bo'lsa n_queries = 150, aks holda 1000.
Xotira guard: har bir konfiguratsiya yakka tartibda ishlaydi, d tarovati
tugagach massivlar bo'shatiladi (8 GB Mac uchun xavfsiz).
"""
from __future__ import annotations

import csv
import gc
import json
import math
import pathlib
import platform
import resource
import sys
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.blocked_memory import BlockedMemory  # noqa: E402
from src.factworld import encode_fact, make_world, query_key  # noqa: E402
from src.fhrr import build_obj_matrix  # noqa: E402

D_VALUES = (512, 1024, 2048, 4096)
B_VALUES = (1, 2, 4, 8, 16, 32)
N_FACTS = 1000
N_SUBJECTS = 50          # make_world default -> kalit fazosi 50 * 20 = 1000
N_RELATIONS = 20
CAP_N_VALUES = (50, 100, 200, 400, 800)
CAP_D_VALUES = (512, 1024, 2048, 4096)
DIAG_D_VALUES = (512, 1024, 2048)   # diagnostika: kalitlari deyarli unikal dunyo
DIAG_B_VALUES = (4, 8, 16, 32)
DIAG_N_SUBJECTS = 1000             # 20 000 kalit -> to'qnashuv ~2%
SWEEP_CSV = ROOT / "results" / "stage3_d_sweep.csv"
CURVE_CSV = ROOT / "results" / "stage3_capacity_curve.csv"
OPTIMAL_JSON = ROOT / "results" / "stage3_optimal_config.json"


# ---------------------------------------------------------------- yordamchilar
def n_queries_for(d: int, b: int) -> int:
    """Runtime guard: og'ir konfiguratsiyalar uchun 150 so'rov."""
    return 150 if (d >= 2048 or b >= 16) else N_FACTS


def query_indices(n_total: int, n_q: int) -> np.ndarray:
    """Deterministik, tekis taqsimlangan indekslar (subsequence)."""
    if n_q >= n_total:
        return np.arange(n_total)
    idx = np.linspace(0, n_total - 1, n_q).astype(int)
    return np.unique(idx)


def peak_rss_mb() -> float:
    """Peak RSS (MB). macOS: bayt, Linux: KB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if platform.system() == "Darwin" else raw / 1e3


def build_cache(d: int, n_facts: int, n_subjects: int) -> dict:
    """Bitta d uchun dunyo + kodkitob + fakt vektorlari.

    B va strategiya bo'ylab qayta ishlatiladi; objects lug'ati mat_n ichida
    takrorlangani uchun bo'shatiladi (xotira guard).
    """
    fw = make_world(n_facts, d, n_subjects=n_subjects, n_relations=N_RELATIONS, seed=0)
    names, mat_n = build_obj_matrix(fw["objects"])
    facts = list(fw["facts"])
    fact_keys = [(s, r) for s, r, _ in facts]
    enc = [encode_fact(fw, s, r, o) for s, r, o in facts]
    keys = [query_key(fw, s, r) for s, r in fact_keys]
    codebook_bytes = sum(v.nbytes for v in fw["objects"].values())
    del fw
    gc.collect()
    return {"d": d, "names": names, "mat_n": mat_n, "enc": enc, "keys": keys,
            "facts": facts, "fact_keys": fact_keys, "key_counts": Counter(fact_keys),
            "codebook_bytes": codebook_bytes,
            "distinct_keys": len(set(fact_keys)), "n_facts": n_facts}


def structural_ceiling(cache: dict, idx: np.ndarray) -> float:
    """Kalit to'qnashuvi bergan kutilayotgan shift (so'rov to'plami bo'yicha).

    Bitta (s, r) kalitga k fakt bog'langan bo'lsa, cleanup faqat bittasini
    qaytaradi -> shu kalitdagi har bir fakt uchun kutilgan aniqlik 1/k.
    """
    counts = cache["key_counts"]
    return float(np.mean([1.0 / counts[cache["fact_keys"][i]] for i in idx]))


def evaluate(cache: dict, b: int, strategy: str, n_q_override: int | None = None) -> dict:
    """Bitta konfiguratsiya: add -> query -> metrikalar."""
    d = cache["d"]
    bm = BlockedMemory(d, n_blocks=b, strategy=strategy, perm=1, seed=0)
    for (s, r, _o), v in zip(cache["facts"], cache["enc"]):
        bm.add(v, s, r)

    n_q_want = n_q_override if n_q_override is not None else n_queries_for(d, b)
    idx = query_indices(cache["n_facts"], n_q_want)
    names, mat_n, facts, keys = cache["names"], cache["mat_n"], cache["facts"], cache["keys"]
    correct = 0
    t0 = time.perf_counter()
    for i in idx:
        name, _sim, _scanned = bm.query(keys[i], names, mat_n)
        correct += name == facts[i][2]
    elapsed = time.perf_counter() - t0

    n_q = len(idx)
    acc = correct / n_q
    mean_count = sum(bm.counts) / b
    row = {
        "d": d, "B": b, "strategy": strategy, "n_queries": n_q,
        "accuracy": round(acc, 4),
        "ms_per_query": round(elapsed / n_q * 1000.0, 4),
        "memory_MB": round((cache["codebook_bytes"] + bm.memory_bytes()) / 1e6, 4),
        "load_ratio": round(max(bm.counts) / mean_count, 4),
        "ci95": round(1.96 * math.sqrt(max(acc * (1.0 - acc), 1e-12) / n_q), 4),
        "ceiling": round(structural_ceiling(cache, idx), 4),
    }
    del bm
    gc.collect()
    return row


def evaluate_capacity_point(d: int, n_facts: int, n_subjects: int) -> dict:
    """B=1 uchun to'liq N bo'yicha accuracy (sig'im egri chizig'i nuqtasi)."""
    cache = build_cache(d, n_facts, n_subjects)
    bm = BlockedMemory(d, n_blocks=1, strategy="round_robin", perm=1, seed=0)
    for (s, r, _o), v in zip(cache["facts"], cache["enc"]):
        bm.add(v, s, r)
    correct = 0
    for i, (s, r, o) in enumerate(cache["facts"]):
        name, _sim, _ = bm.query(cache["keys"][i], cache["names"], cache["mat_n"])
        correct += name == o
    row = {"d": d, "B": 1, "N": n_facts, "accuracy": round(correct / n_facts, 4),
           "distinct_keys": cache["distinct_keys"], "ceiling": round(cache["distinct_keys"] / n_facts, 4)}
    del bm, cache
    gc.collect()
    return row


def formula_check(curve_rows: list[dict], C: int = 1000) -> list[dict]:
    """Empirik N90 (accuracy < 0.9 bo'ladigan chegara) vs nazariy M_cap.

    N90 = acc(N) = 0.9 kesishmasi; qo'shni nuqtalar orasida log(N) bo'yicha
    chiziqli interpolyatsiya. Qavs [oxirgi acc>=0.9, birinchi acc<0.9].
    """
    out = []
    for d in sorted({r["d"] for r in curve_rows}):
        pts = sorted((r["N"], r["accuracy"]) for r in curve_rows if r["d"] == d)
        pred = d / (2.0 * math.log(C))
        pass_n = [n for n, a in pts if a >= 0.9]
        fail_n = [n for n, a in pts if a < 0.9]
        lo = max(pass_n) if pass_n else None
        hi = min(fail_n) if fail_n else None
        n90 = None
        if lo is not None and hi is not None and hi > lo:
            a_lo = dict(pts)[lo]
            a_hi = dict(pts)[hi]
            frac = (a_lo - 0.9) / (a_lo - a_hi) if a_lo != a_hi else 0.0
            frac = min(max(frac, 0.0), 1.0)
            n90 = math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        ratio = n90 / pred if n90 is not None else None
        verdict = "mos keldi" if (ratio is not None and abs(ratio - 1.0) <= 0.20) else "kelmadi"
        out.append({"d": d, "predicted_Mcap": round(pred, 1),
                    "empirical_N90": None if n90 is None else round(n90, 1),
                    "bracket": (lo, hi), "ratio": None if ratio is None else round(ratio, 2),
                    "verdict": verdict})
    return out


def print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    print("  ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(header, widths))))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))))


def write_csv(path: pathlib.Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------- asosiy
def main() -> None:
    t_start = time.perf_counter()
    print(f"Stage 3 — capacity engineering | N={N_FACTS} | codebook C={N_FACTS} | "
          f"world key space {N_SUBJECTS}x{N_RELATIONS}={N_SUBJECTS * N_RELATIONS}")

    sweep: list[dict] = []
    best_by_d: dict[int, float] = {}

    # 1-faza: round_robin, barcha d (bir vaqtda bitta d xotirada)
    for d in D_VALUES:
        cache = build_cache(d, N_FACTS, N_SUBJECTS)
        print(f"\nd={d}: {cache['distinct_keys']}/{N_FACTS} unikal kalit "
              f"-> shift {cache['distinct_keys'] / N_FACTS:.3f}")
        best = 0.0
        for b in B_VALUES:
            row = evaluate(cache, b, "round_robin")
            sweep.append(row)
            best = max(best, row["accuracy"])
            print(f"  B={b:<3} nq={row['n_queries']:<5} acc={row['accuracy']:.3f} "
                  f"ms/q={row['ms_per_query']:.3f} mem={row['memory_MB']:.2f}MB "
                  f"load={row['load_ratio']:.3f}")
            if peak_rss_mb() > 4500:
                raise MemoryError("RSS 4.5 GB dan oshdi — sweep to'xtatildi")
        best_by_d[d] = best
        del cache
        gc.collect()

    # 2-faza: similarity strategiyasi — eng yaxshi 2 ta d uchun
    top2 = sorted(best_by_d, key=lambda k: (-best_by_d[k], k))[:2]
    print(f"\nTop-2 d (round_robin bo'yicha): {top2}")
    for d in top2:
        cache = build_cache(d, N_FACTS, N_SUBJECTS)
        for b in B_VALUES:
            row = evaluate(cache, b, "similarity")
            sweep.append(row)
            print(f"  d={d} B={b:<3} similarity acc={row['accuracy']:.3f} "
                  f"ms/q={row['ms_per_query']:.3f} load={row['load_ratio']:.3f}")
        del cache
        gc.collect()

    sweep.sort(key=lambda r: (r["d"], r["strategy"], r["B"]))
    write_csv(SWEEP_CSV, ["d", "B", "strategy", "n_queries", "accuracy",
                          "ms_per_query", "memory_MB", "load_ratio", "ci95", "ceiling"], sweep)

    # optimal konfiguratsiya: eng yaxshi 5 nomzodni TO'LIQ N=1000 da tasdiqlash
    # (150 so'rovli baho shovqinli: ~+-0.07; CSV'da guard bo'yicha qoladi, JSON'da
    #  to'liq N natijasi yoziladi)
    cands = sorted(sweep, key=lambda r: (-r["accuracy"], r["memory_MB"],
                                         r["ms_per_query"]))[:5]
    confirm: list[dict] = []
    cur_d = None
    cache_d = None
    for cand in cands:
        if cand["d"] != cur_d:
            del cache_d
            gc.collect()
            cache_d = build_cache(cand["d"], N_FACTS, N_SUBJECTS)
            cur_d = cand["d"]
        row = evaluate(cache_d, cand["B"], cand["strategy"], n_q_override=N_FACTS)
        row["acc_subset"] = cand["accuracy"]
        row["n_queries_subset"] = cand["n_queries"]
        confirm.append(row)
        print(f"  confirm d={row['d']} B={row['B']} {row['strategy']}: "
              f"subset {cand['accuracy']:.3f} -> full-N {row['accuracy']:.3f}")
    del cache_d
    gc.collect()

    best = sorted(confirm, key=lambda r: (-r["accuracy"], r["memory_MB"],
                                           r["ms_per_query"]))[0]
    target_met = best["accuracy"] >= 0.90
    optimal = {
        "d": best["d"], "B": best["B"], "strategy": best["strategy"],
        "acc": best["accuracy"], "ms_per_query": best["ms_per_query"],
        "memory_MB": best["memory_MB"],
        "target_met": bool(target_met),
        "best_acc": best["accuracy"],
        "n_queries": best["n_queries"],
        "ci95": best["ci95"],
        "ceiling": best["ceiling"],
        "N": N_FACTS,
        "n_subjects": N_SUBJECTS,
        "acc_sweep_subset": best["acc_subset"],
        "n_queries_sweep": best["n_queries_subset"],
        "note": ("accuracy shiftga urildi: N=1000 fakt 50x20=1000 kalit fazosida "
                 "~37% takroriy (s,r) kalit beradi; bitta kalitga bir nechta obyekt "
                 "biriktirilgani uchun cleanup faqat bittasini qaytara oladi. "
                 f"Shu dunyo uchun strukturaviy shift = {best['ceiling']:.3f}. "
                 "acc = to'liq N=1000 bo'yicha tasdiqlangan aniqlik; "
                 "acc_sweep_subset = CSV'dagi (150 so'rovli) baho."),
    }
    OPTIMAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    OPTIMAL_JSON.write_text(json.dumps(optimal, indent=2) + "\n")

    # 3-faza: sig'im egri chizig'i (B=1), d bo'yicha
    curve: list[dict] = []
    for d in CAP_D_VALUES:
        for n in CAP_N_VALUES:
            row = evaluate_capacity_point(d, n, N_SUBJECTS)
            curve.append(row)
            print(f"  capacity d={d} N={n:<4} acc={row['accuracy']:.3f} "
                  f"(shift {row['ceiling']:.3f})")
        gc.collect()
    write_csv(CURVE_CSV, ["d", "B", "N", "accuracy"], curve)

    # 4-faza: nazariy formula tekshiruvi
    checks = formula_check(curve)

    # 5-faza: diagnostika — kalitlari deyarli unikal dunyo (n_subjects=1000)
    diag: list[dict] = []
    for d in DIAG_D_VALUES:
        cache = build_cache(d, N_FACTS, DIAG_N_SUBJECTS)
        for b in DIAG_B_VALUES:
            row = evaluate(cache, b, "round_robin")
            diag.append(row)
        del cache
        gc.collect()
    # diagnostikada "sig'im o'qi" — blokdagi faktlar soni M = N/B
    diag_checks = formula_check([{"d": r["d"], "N": N_FACTS // r["B"], "accuracy": r["accuracy"]}
                                 for r in diag])

    # ------------------------------------------------------------------ jadvallar
    print_table("SWEEP (results/stage3_d_sweep.csv)",
                ["d", "B", "strategy", "nq", "acc", "ms/q", "MB", "load", "ceiling"],
                [[str(r["d"]), str(r["B"]), r["strategy"], str(r["n_queries"]),
                  f"{r['accuracy']:.3f}", f"{r['ms_per_query']:.3f}",
                  f"{r['memory_MB']:.2f}", f"{r['load_ratio']:.2f}", f"{r['ceiling']:.3f}"]
                 for r in sweep])

    print_table("CAPACITY CURVE, B=1 (results/stage3_capacity_curve.csv)",
                ["d", "N", "acc", "shift"],
                [[str(r["d"]), str(r["N"]), f"{r['accuracy']:.3f}", f"{r['ceiling']:.3f}"]
                 for r in curve])

    print_table("FORMULA: M_cap ~ d/(2 ln C), C=1000",
                ["d", "M_cap_pred", "N90_emp", "qavs", "ratio", "verdict"],
                [[str(c["d"]), f"{c['predicted_Mcap']:.1f}",
                  "-" if c["empirical_N90"] is None else f"{c['empirical_N90']:.1f}",
                  str(c["bracket"]), "-" if c["ratio"] is None else f"{c['ratio']:.2f}",
                  c["verdict"]] for c in checks])

    print_table("DIAGNOSTIKA: n_subjects=1000 (kalitlar deyarli unikal), nq=150",
                ["d", "B", "M=N/B", "acc", "shift"],
                [[str(r["d"]), str(r["B"]), f"{N_FACTS / r['B']:.0f}", f"{r['accuracy']:.3f}",
                  f"{r['ceiling']:.3f}"] for r in diag])

    print_table("DIAGNOSTIKA FORMULA: M o'qi = N/B",
                ["d", "M_cap_pred", "M90_emp", "qavs", "ratio", "verdict"],
                [[str(c["d"]), f"{c['predicted_Mcap']:.1f}",
                  "-" if c["empirical_N90"] is None else f"{c['empirical_N90']:.1f}",
                  str(c["bracket"]), "-" if c["ratio"] is None else f"{c['ratio']:.2f}",
                  c["verdict"]] for c in diag_checks])

    print_table("TASDIQLASH: eng yaxshi 5 nomzod, to'liq N=1000",
                ["d", "B", "strategy", "acc@subset", "nq_sub", "acc@full", "ms/q", "MB"],
                [[str(r["d"]), str(r["B"]), r["strategy"], f"{r['acc_subset']:.3f}",
                  str(r["n_queries_subset"]), f"{r['accuracy']:.3f}",
                  f"{r['ms_per_query']:.3f}", f"{r['memory_MB']:.2f}"] for r in confirm])

    print(f"\nOPTIMAL: d={optimal['d']} B={optimal['B']} {optimal['strategy']} "
          f"acc={optimal['acc']:.3f} (target_met={optimal['target_met']}) "
          f"ms/q={optimal['ms_per_query']:.3f} mem={optimal['memory_MB']:.2f}MB")

    # markdown eksport (docs/capacity_formula.md uchun)
    print("\n--- markdown: sweep ---")
    print("| d | B | strategy | n_queries | acc | ms/q | MB | load |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sweep:
        print(f"| {r['d']} | {r['B']} | {r['strategy']} | {r['n_queries']} | "
              f"{r['accuracy']:.3f} | {r['ms_per_query']:.3f} | {r['memory_MB']:.2f} | "
              f"{r['load_ratio']:.2f} |")
    print("\n--- markdown: capacity ---")
    print("| d | N=50 | N=100 | N=200 | N=400 | N=800 |")
    print("|---|---|---|---|---|---|")
    for d in CAP_D_VALUES:
        cells = {r["N"]: r["accuracy"] for r in curve if r["d"] == d}
        print(f"| {d} | " + " | ".join(f"{cells[n]:.3f}" for n in CAP_N_VALUES) + " |")
    print("\n--- markdown: formula ---")
    print("| d | M_cap nazariy | N90 empirik | qavs | ratio | xulosa |")
    print("|---|---|---|---|---|---|")
    for c in checks:
        n90 = "-" if c["empirical_N90"] is None else f"{c['empirical_N90']:.0f}"
        ratio = "-" if c["ratio"] is None else f"{c['ratio']:.2f}"
        print(f"| {c['d']} | {c['predicted_Mcap']:.0f} | {n90} | {c['bracket']} | "
              f"{ratio} | {c['verdict']} |")
    print("\n--- markdown: diagnostika ---")
    print("| d | B | M=N/B | acc |")
    print("|---|---|---|---|")
    for r in diag:
        print(f"| {r['d']} | {r['B']} | {N_FACTS / r['B']:.0f} | {r['accuracy']:.3f} |")
    print("\n--- markdown: diagnostika formula ---")
    print("| d | M_cap nazariy | M90 empirik | qavs | ratio | xulosa |")
    print("|---|---|---|---|---|---|")
    for c in diag_checks:
        m90 = "-" if c["empirical_N90"] is None else f"{c['empirical_N90']:.0f}"
        ratio = "-" if c["ratio"] is None else f"{c['ratio']:.2f}"
        print(f"| {c['d']} | {c['predicted_Mcap']:.0f} | {m90} | {c['bracket']} | "
              f"{ratio} | {c['verdict']} |")

    print(f"\nRuntime: {time.perf_counter() - t_start:.1f}s | peak RSS {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()

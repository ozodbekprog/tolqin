"""Stage 2 — blok masshtablash benchmark (blocked memory scaling).

Ishga tushirish (repo ildizidan):
    python3 scripts/stage2_scaling.py

Natija: results/stage2_scaling.csv

CSV ustunlari (CONTRACT.md talabi):
  d           : vektor o'lchami (512 yoki 1024)
  B           : bloklar soni
  strategy    : "round_robin" | "hash" | "similarity"
                (B=1 da strategiya ahamiyatsiz -> "round_robin" yoziladi)
  query_mode  : "all" (top_k=None, early_stop=None) yoki "top_k=K" yoki
                "early_stop=T" — bular faqat B=8 round_robin holatida o'lchanadi
  n_queries   : shu qatorda o'lchangan query soni (1000 yoki 200, agar
                config 60s dan oshib ketish xavfi bo'lsa 200 ga tushiriladi;
                indekslar butun to'plam bo'ylab tekis tanlanadi)
  accuracy    : to'g'ri cleanup topilgan querylar ulushi [0,1]
  query_ms    : o'lchangan querylar uchun devor vaqti (ms). Har bir query
                uchun 3 ta bir xil o'tishdagi eng tez vaqt olinadi va
                yig'iladi (GC o'chirilgan; fon yuklamasi ta'sirini
                kamaytirish uchun). Bu min-of-3 baho — biroz pastga siljigan.
  ms_per_query: query_ms / n_queries
  memory_MB   : bm.memory_bytes()/1e6 (faqat blok vektorlari; Python
                objekt overhead'i hisobga olinmagan)
  load_ratio  : max(counts)/min(counts) bo'sh bo'lmagan bloklar bo'yicha;
                bo'sh blok bo'lsa "inf" (round_robin har doim 1.00 — u
                konstruksiya bo'yicha mukammal balanslangan)

Eslatma: accuracy shifti make_world() da (s,r) kalitlari takrorlanishi bilan
cheklangan (1000 fakt / 50 subyekt x 20 munosabat = 1000 kombinatsiya,
amalda ~62% unikal). Shu sababli yuqori B da accuracy ~0.62 da to'xtaydi;
skript buni dunyo sarlavhasida ("unique(s,r) ceiling") chop etadi.

Outlierlar: query narxi skanerlangan bloklar soniga chiziqli, shuning uchun
o'lchov B=1 narxidan kutilganidan 1.6x katta bo'lsa konfiguratsiya qayta
o'lchanadi (eng tez urinish saqlanadi) — bu fon yuklamasi shovqinini
kamaytiradi.
"""
from __future__ import annotations

import csv
import gc
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.blocked_memory import BlockedMemory  # noqa: E402
from src.factworld import encode_fact, make_world, query_key  # noqa: E402
from src.fhrr import build_obj_matrix  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "stage2_scaling.csv"

WORLDS = ((512, 1), (1024, 2))
B_LIST = (1, 2, 4, 8, 16, 32)
STRATEGIES = ("round_robin", "hash", "similarity")
# (query_mode nomi, top_k, early_stop)
QUERY_MODES = (
    ("all", None, None),
    ("top_k=2", 2, None),
    ("top_k=4", 4, None),
    ("top_k=8", 8, None),
    ("early_stop=0.9", None, 0.9),
)
MAX_CONFIG_SECONDS = 60.0
SUB_QUERIES = 200
FIELDS = ["d", "B", "strategy", "query_mode", "n_queries", "accuracy",
          "query_ms", "ms_per_query", "memory_MB", "load_ratio"]


def even_indices(n_total: int, n_take: int) -> list[int]:
    """Butun to'plam bo'ylab tekis taqsimlangan indekslar."""
    if n_take >= n_total:
        return list(range(n_total))
    step = n_total / n_take
    return [int(i * step) for i in range(n_take)]


def load_ratio(counts: list[int]) -> float | str:
    used = [c for c in counts if c > 0]
    if len(used) < len(counts):
        return "inf"
    return max(counts) / min(counts)


def build_memory(d: int, fvecs, facts, B: int, strategy: str) -> BlockedMemory:
    bm = BlockedMemory(d, n_blocks=B, strategy=strategy, perm=1, seed=0)
    for fvec, (s, r, _) in zip(fvecs, facts):
        bm.add(fvec, subject=s, relation=r)
    return bm


def run_queries(bm, keys, targets, names, mat_n, idx, top_k, early_stop,
                timing_reps: int = 3):
    """(accuracy, total_ms, mean_scanned_blocks, max_best_sim).

    accuracy birinchi to'liq o'tishdan olinadi (deterministik).
    query_ms = har bir query uchun `timing_reps` ta o'tishdagi eng tez vaqt
    yig'indisi — mashinada fon yuklamasi bo'lsa ham barqarorroq o'lchov
    (bu pastga qarab siljigan baho: min-of-N).
    max_best_sim — eng yuqori cleanup o'xshashligi (early_stop chegarasi
    qachon yetib borishini ko'rsatish uchun).
    """
    per_query_s = None
    correct = 0
    scanned = 0
    max_sim = -2.0
    gc.disable()  # vaqt o'lchashda GC pauzalarini olib tashlash
    try:
        for rep in range(timing_reps):
            pass_s = []
            p_correct = 0
            p_scanned = 0
            for i in idx:
                t0 = time.perf_counter()
                name, sim, n_scan = bm.query(keys[i], names, mat_n, perm_used=1,
                                             top_k=top_k, early_stop=early_stop)
                pass_s.append(time.perf_counter() - t0)
                p_correct += name == targets[i]
                p_scanned += n_scan
                max_sim = max(max_sim, sim)
            if rep == 0:
                per_query_s = pass_s
                correct, scanned = p_correct, p_scanned
            else:
                per_query_s = [min(a, b) for a, b in zip(per_query_s, pass_s)]
    finally:
        gc.enable()
    total_ms = sum(per_query_s) * 1000.0
    return correct / len(idx), total_ms, scanned / len(idx), max_sim


def pick_n_queries(bm, keys, names, mat_n, n_total: int,
                   timing_reps: int = 3) -> int:
    """Kalibrlash: 8 query bo'yicha 1000 query vaqtini baholaydi."""
    probe = even_indices(n_total, min(8, n_total))
    t0 = time.perf_counter()
    for i in probe:
        bm.query(keys[i], names, mat_n, perm_used=1)
    est_s = (time.perf_counter() - t0) * (n_total / len(probe)) * timing_reps
    return n_total if est_s <= MAX_CONFIG_SECONDS else SUB_QUERIES


def eval_config(d, fvecs, facts, targets, keys, names, mat_n, B, strategy,
                query_mode="all", top_k=None, early_stop=None,
                unit_ms=None, max_tries=3):
    """Bitta konfiguratsiyani o'lchaydi.

    `unit_ms` — bitta blokni skanerlashning kutilgan narxi (B=1 o'lchovidan).
    Query narxi skanerlangan bloklar soniga chiziqli, shuning uchun o'lchov
    `1.6 * unit_ms * scanned` dan sezilarli katta bo'lsa — bu fon yuklamasi
    (outlier) — qayta o'lchanadi; eng tez urinish saqlanadi.
    """
    bm = build_memory(d, fvecs, facts, B, strategy)
    n_q = pick_n_queries(bm, keys, names, mat_n, len(facts))
    idx = even_indices(len(facts), n_q)
    for i in idx[:8]:  # warmup: shu query rejimining kod yo'lini isitish
        bm.query(keys[i], names, mat_n, perm_used=1, top_k=top_k,
                 early_stop=early_stop)
    best = None
    for _attempt in range(max_tries):
        acc, q_ms, scanned, max_sim = run_queries(bm, keys, targets, names,
                                                  mat_n, idx, top_k, early_stop)
        lr = load_ratio(bm.counts)
        row = {
            "d": d,
            "B": B,
            "strategy": strategy,
            "query_mode": query_mode,
            "n_queries": len(idx),
            "accuracy": round(acc, 4),
            "query_ms": round(q_ms, 1),
            "ms_per_query": round(q_ms / len(idx), 4),
            "memory_MB": round(bm.memory_bytes() / 1e6, 4),
            "load_ratio": lr if isinstance(lr, str) else round(lr, 3),
        }
        if best is None or row["ms_per_query"] < best[0]["ms_per_query"]:
            best = (row, scanned, max_sim)
        if unit_ms is None:
            break
        expected = unit_ms * max(scanned, 1.0)
        if row["ms_per_query"] <= 1.6 * expected:
            break
        print(f"    [retry] d={d} B={B} {strategy} {query_mode}: "
              f"{row['ms_per_query']:.3f} ms/q >> expected {expected:.3f} "
              f"(background load?)")
    return best


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.perf_counter()

    for d, seed in WORLDS:
        t_world = time.perf_counter()
        fw = make_world(n_facts=1000, d=d, seed=seed)
        names, mat_n = build_obj_matrix(fw["objects"])
        targets = [o for (_s, _r, o) in fw["facts"]]
        keys = [query_key(fw, s, r) for (s, r, _o) in fw["facts"]]
        fvecs = [encode_fact(fw, s, r, o) for (s, r, o) in fw["facts"]]
        ceiling = len({(s, r) for s, r, _o in fw["facts"]}) / len(fw["facts"])

        print(f"\n=== d={d}  N={len(fw['facts'])}  objects={len(names)}  "
              f"unique(s,r) ceiling={ceiling:.3f} ===")
        print(f"{'B':>3} {'strategy':<12} {'acc':>6} {'ms/query':>9} "
              f"{'mem_MB':>7} {'load':>6}")
        # B=1 birinchi o'lchanadi: uning ms/query'si "bitta blok" narxi (unit_ms)
        unit_ms = None
        for B in B_LIST:
            strategies = ("round_robin",) if B == 1 else STRATEGIES
            for strategy in strategies:
                row, _scanned, _msim = eval_config(d, fvecs, fw["facts"], targets,
                                                   keys, names, mat_n, B, strategy,
                                                   unit_ms=unit_ms)
                rows.append(row)
                if B == 1:
                    unit_ms = row["ms_per_query"]
                print(f"{B:>3} {strategy:<12} {row['accuracy']:>6.3f} "
                      f"{row['ms_per_query']:>9.3f} {row['memory_MB']:>7.3f} "
                      f"{str(row['load_ratio']):>6}")

        # --- query rejimlari: B=8 round_robin (eng yaxshi blok konfiguratsiyasi) ---
        print(f"\n-- d={d}: query modes on B=8 round_robin (full-scan acc baseline "
              f"above) --")
        print(f"{'query_mode':<16} {'acc':>6} {'scanned':>8} {'ms/query':>9} "
              f"{'max_sim':>7}")
        for mode_name, top_k, early_stop in QUERY_MODES:
            row, scanned, max_sim = eval_config(d, fvecs, fw["facts"], targets,
                                                keys, names, mat_n, 8,
                                                "round_robin",
                                                query_mode=mode_name, top_k=top_k,
                                                early_stop=early_stop,
                                                unit_ms=unit_ms)
            rows.append(row)
            print(f"{mode_name:<16} {row['accuracy']:>6.3f} {scanned:>8.2f} "
                  f"{row['ms_per_query']:>9.3f} {max_sim:>7.3f}")
        print(f"[d={d} world done in {time.perf_counter() - t_world:.1f}s]")

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows) "
          f"in {time.perf_counter() - t_start:.1f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

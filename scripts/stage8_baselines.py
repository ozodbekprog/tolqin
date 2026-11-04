"""Stage 8 — VSA vs hash: to'liq/qisman/shovqinli kalit taqqoslash.

Dunyo: 200 subject, har biriga 5 object bog'langan (N=1000 juftlik, C=1000 object).
VSA: juftlik kodlash f = bind(S, permute(O,1)); bloklash hash(subject) bo'yicha
     (bir subjectning hamma objecti bitta blokda).
Baseline'lar: (a) HashKV — faqat to'liq kalit; (b) InvertedIndex — s -> {o}.

Metrikalar: recall@5, precision@5, top-1 hit, ms/so'rov, xotira.
"""
from __future__ import annotations

import sys, pathlib, time, csv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from src.fhrr import build_obj_matrix, bind, permute, inv_permute, random_fhrr, unbind
from src.blocked_memory import BlockedMemory

SUBJECTS = 200
LINKS = 5
DEFAULT_D = 1024
SEED = 7


def build_world(d, seed=SEED):
    rng = np.random.default_rng(seed)
    subj = {f"s{i}": random_fhrr(d, rng) for i in range(SUBJECTS)}
    obj = {f"o{i}": random_fhrr(d, rng) for i in range(SUBJECTS * LINKS)}
    links = {f"s{i}": [f"o{i * LINKS + j}" for j in range(LINKS)] for i in range(SUBJECTS)}
    facts = [(s, o) for s, objs in links.items() for o in objs]
    return subj, obj, links, facts


class HashKV:
    """Oddiy kalit-qiymat hash: faqat to'liq (s,o) kalit."""
    def __init__(self):
        self.d = {}

    def store(self, s, o):
        self.d[(s, o)] = o

    def query(self, s, o):
        return self.d.get((s, o))

    def memory_bytes(self):
        return sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.d.items()) + sys.getsizeof(self.d)


class InvertedIndex:
    """Adolatli baseline: subject -> objectlar ro'yxati."""
    def __init__(self):
        self.d = {}

    def store(self, s, o):
        self.d.setdefault(s, []).append(o)

    def query(self, s):
        return list(self.d.get(s, []))

    def memory_bytes(self):
        return (sum(sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v)
                    for k, v in self.d.items()) + sys.getsizeof(self.d))


def vsa_scores(bm, s_vec, perm=1):
    """Barcha bloklar bo'yicha candidate vektor -> nomlar, mat_n (tashqarida)."""
    vecs = [inv_permute(unbind(m, s_vec), perm) for m, c in zip(bm.blocks, bm.counts) if c]
    return vecs


def topk_from_blocks(vecs, names, mat_n, k):
    sims = None
    for v in vecs:
        vn = v / (np.linalg.norm(v) + 1e-12)
        s = np.real(mat_n @ np.conj(vn))
        sims = s if sims is None else np.maximum(sims, s)
    idx = np.argsort(sims)[::-1][:k]
    return [(names[i], float(sims[i])) for i in idx]


def noisy(v, level, rng):
    if level <= 0:
        return v
    n = (rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)) * level / np.sqrt(2)
    out = v + n.astype(np.complex64)
    return out / (np.linalg.norm(out) + 1e-12)


def main():
    rows = []
    d = DEFAULT_D
    subj, obj, links, facts = build_world(d)
    names, mat_n = build_obj_matrix(obj)
    fvecs = [bind(subj[s], permute(obj[o], 1)) for s, o in facts]

    # --- hash baselinelar
    hk = HashKV(); inv = InvertedIndex()
    t0 = time.perf_counter()
    for s, o in facts:
        hk.store(s, o); inv.store(s, o)
    build_ms_hash = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for s, _ in facts:
        inv.query(s)
    inv_ms = (time.perf_counter() - t0) * 1000 / len(facts)

    # qisman kalit: inverted index
    rec, prec = [], []
    t0 = time.perf_counter()
    for s in subj:
        got = inv.query(s)
        true = set(links[s])
        rec.append(len(set(got) & true) / len(true))
        prec.append(len(set(got) & true) / max(len(got), 1))
    inv_q_ms = (time.perf_counter() - t0) * 1000 / len(subj)

    rows.append(dict(task="partial_cue", method="inverted_index", d=d, B=0,
                     recall=float(np.mean(rec)), precision=float(np.mean(prec)), top1_hit=1.0,
                     ms_per_query=inv_q_ms, memory_MB=inv.memory_bytes() / 1e6))
    rows.append(dict(task="partial_cue", method="hash_kv", d=d, B=0,
                     recall=0.0, precision=0.0, top1_hit=0.0,
                     ms_per_query=0.0, memory_MB=hk.memory_bytes() / 1e6,
                     note="faqat to'liq kalit; qisman so'rovga qodir emas"))

    # --- VSA bloklari
    for B in (1, 8, 32):
        bm = BlockedMemory(d, B, "hash", perm=1)
        t0 = time.perf_counter()
        for (s, o), fv in zip(facts, fvecs):
            bm.add(fv, s, "pair")
        build_ms = (time.perf_counter() - t0) * 1000

        rec, prec, hit = [], [], []
        t0 = time.perf_counter()
        for s in subj:
            vecs = vsa_scores(bm, subj[s])
            top = topk_from_blocks(vecs, names, mat_n, LINKS)
            got = [n for n, _ in top]
            true = set(links[s])
            rec.append(len(set(got) & true) / len(true))
            prec.append(len(set(got) & true) / len(got))
            hit.append(1.0 if got[0] in true else 0.0)
        q_ms = (time.perf_counter() - t0) * 1000 / len(subj)
        rows.append(dict(task="partial_cue", method=f"vsa_fhrr", d=d, B=B,
                         recall=float(np.mean(rec)), precision=float(np.mean(prec)),
                         top1_hit=float(np.mean(hit)), ms_per_query=q_ms,
                         memory_MB=(bm.memory_bytes() + mat_n.nbytes) / 1e6))

    # --- shovqinli kalit
    rng = np.random.default_rng(99)
    from src.fhrr import similarity as _sim
    cue_sims = {lv: float(np.mean([_sim(subj[s], noisy(subj[s], lv, np.random.default_rng(1000 + i)))
                                   for i, s in enumerate(subj)])) for lv in (0.1, 0.3, 0.5, 1.0, 2.0)}
    for level in (0.1, 0.3, 0.5, 1.0, 2.0):
        # baseline: aniq kalit izlash muvaffaqiyatsiz (kalit o'zgargan)
        rows.append(dict(task=f"noisy_cue_{int(level*100)}pct", method="exact_lookup", d=d, B=0,
                         recall=0.0, precision=0.0, top1_hit=0.0, ms_per_query=0.0,
                         memory_MB=hk.memory_bytes() / 1e6,
                         note=f"kalit mos kelmaydi -> miss; cue_sim={cue_sims[level]:.3f}"))
        for B in (8, 32):
            bm = BlockedMemory(d, B, "hash", perm=1)
            for (s, o), fv in zip(facts, fvecs):
                bm.add(fv, s, "pair")
            true_hits = 0
            t0 = time.perf_counter()
            nq = 0
            for s in subj:
                q = noisy(subj[s], level, rng)
                # qaysi subjectga tegishli ekanini topish: bloklar bo'yicha top-1 object
                vecs = vsa_scores(bm, q)
                top = topk_from_blocks(vecs, names, mat_n, 1)
                if top[0][0] in set(links[s]):
                    true_hits += 1
                nq += 1
            q_ms = (time.perf_counter() - t0) * 1000 / nq
            rows.append(dict(task=f"noisy_cue_{int(level*100)}pct", method="vsa_fhrr", d=d, B=B,
                             recall=true_hits / nq, precision="", top1_hit=true_hits / nq,
                             ms_per_query=q_ms, memory_MB=(bm.memory_bytes() + mat_n.nbytes) / 1e6,
                             note=f"cue_sim={cue_sims[level]:.3f}"))

    out = pathlib.Path("results/stage8_baselines.csv")
    fields = ["task", "method", "d", "B", "recall", "precision", "top1_hit", "ms_per_query", "memory_MB", "note"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    for r in rows:
        prec = f"{float(r['precision']):.3f}" if r['precision'] != "" else "  -  "
        print(f"{r['task']:18s} {r['method']:14s} B={r['B']:<3} "
              f"recall={float(r['recall']):.3f} prec={prec} "
              f"top1={float(r['top1_hit']):.3f} {float(r['ms_per_query']):.4f}ms "
              f"{float(r['memory_MB']):.2f}MB")
    print("\nwrote", out)


if __name__ == "__main__":
    main()

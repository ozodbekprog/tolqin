# To'lqin / VSA — modul kontrakti (FROZEN)

Barcha modullar shu API'ga qat'iy rioya qiladi. `src/fhrr.py`, `src/factworld.py`,
`src/blocked_memory.py` — **muzlatilgan**, faqat egasi (asosiy agent) o'zgartiradi.
Har bir skript boshida:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
```

va `from src.fhrr import ...` ko'rinishida import qiladi. Skriptlar repo ildizidan
`python3 scripts/xxx.py` bilan ishlaydi.

## src/fhrr.py (muzlatilgan)

```python
random_fhrr(d, rng=None) -> np.complex64[d]      # unit magnitude
bind(a, b) -> np.complex64[d]                    # a * b
unbind(c, b) -> np.complex64[d]                  # c * conj(b)
permute(v, k=1) / inv_permute(v, k=1)            # np.roll
similarity(a, b) -> float                        # real cosine, [-1, 1]
cleanup_proper(r, codebook: dict[str, vec]) -> (name, sim)
cleanup_matrix(r, names: list[str], mat_n: np.ndarray) -> (name, sim, sims)
build_obj_matrix(codebook: dict) -> (names, mat_n)   # mat_n: (C,d), rows = v/sqrt(d)
```

## src/factworld.py (muzlatilgan)

```python
make_world(n_facts=1000, d=512, n_subjects=50, n_relations=20, seed=0) -> dict
# {"subjects": {name: vec}, "relations": {...}, "objects": {...}, "facts": [(s, r, o), ...]}
encode_fact(fw, s, r, o, perm=1) -> vec           # bind(bind(S,R), permute(O))
query_key(fw, s, r) -> vec                        # bind(S, R)
retrieve(fw, M, s, r, perm=1) -> vec              # inv_permute(unbind(unbind(M,S),R))
query_all(fw, M, perm=1) -> list[(name, sim)]     # hamma fakt bo'yicha cleanup natijasi
```

## src/blocked_memory.py (muzlatilgan)

```python
bm = BlockedMemory(d, n_blocks=1, strategy="round_robin", perm=1, seed=0)
# strategy: "round_robin" | "hash" | "similarity"
bm.add(fvec, subject=None, relation=None) -> int           # qaysi blokka tushdi
bm.query(key, names, mat_n, perm_used=1, top_k=None, early_stop=None)
    -> (name, sim, n_blocks_scanned)
bm.memory_bytes() -> int
bm.blocks -> list[np.ndarray]
```

## Natija fayllari

- `results/baseline_results.csv` (stage0)
- `results/stage2_scaling.csv`
- `results/stage3_d_sweep.csv`, `results/stage3_optimal_config.json`
- `results/stage4_dense_vs_sparse.csv`
- `results/stage5_factorization.csv`
- `results/stage6_forgetting_curve.csv`
- `results/final_benchmark.csv`, `results/ablation.csv`, `results/scaling_curves.png`
- `docs/capacity_formula.md`, `docs/FINAL_REPORT.md`

Aniqlik = to'g'ri cleanup topilgan faktlar ulushi (accuracy ∈ [0,1]).
Vaqt = ms (per-query o'rtacha yoki jami, ustun nomida aniq yozilsin).
Xotira = MB. Har CSV ustun nomlari skript boshida hujjatlashtiriladi.

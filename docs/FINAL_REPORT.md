# To'lqin (FHRR/VSA xotira) — yakuniy hisobot

**Sana:** 2026-10-29 · **Muhit:** MacBook Air M3, 8 GB, Python 3.12, NumPy 2.5 (boshqa kutubxonalarsiz)
**Repozitoriy:** `~/Documents/tolqin` · **Holat:** 0–7 bosqichlar bajarildi, barcha raqamlar o'lchangan

## 1. Nima qurildi

FHRR (Fourier Holographic Reduced Representation) asosidagi assotsiativ xotira:
unit-magnitudali kompleks vektorlar, faza qo'shish/ayirish orqali bind/unbind,
blokli xotira, kapasitet injeneriyasi, sparse VSA, resonator, FSRS unutish + LP
e'tibor qatlami. Har bosqich alohida skript va CSV beradi.

| Bosqich | Modul / skript | Holat |
|---|---|---|
| 0. Baseline (eski HRR) | `src/baseline_hrr.py`, `scripts/stage0_baseline.py` | ✅ |
| 1. FHRR yadro | `src/fhrr.py`, `src/factworld.py` | ✅ unit testlar |
| 2. Blokli xotira | `src/blocked_memory.py`, `scripts/stage2_scaling.py` | ✅ |
| 3. Kapasitet | `scripts/stage3_capacity.py`, `docs/capacity_formula.md` | ✅ 90%+ topildi |
| 4. Sparse VSA | `src/sparse_vsa.py`, `scripts/stage4_dense_vs_sparse.py` | ⚠️ ishlamadi |
| 5. Resonator | `src/resonator.py`, `scripts/stage5_factorization.py` | ✅ qismiy |
| 6. FSRS + LP | `src/fsrs.py`, `src/lp_attention.py`, `src/integrated_memory.py` | ✅ |
| 7. Yakuniy benchmark | `scripts/stage7_benchmark.py`, `results/final_benchmark.csv` | ✅ |

Testlar: `python3 -m unittest discover -s tests -v` — 42 test o'tdi, 1 skip.

> **Vaqt o'lchovlari haqida:** skriptlar parallel ishlaganda CPU bo'linishi tufayli
> vaqt 2–4× shovqinli chiqqan. Yakuniy toza o'lchov (izolyatsiyada, 1000 so'rov):
> d=1024/B=32 → **1.87 ms/so'rov**, d=512/B=32 → 1.23 ms, d=2048/B=8 → 1.18 ms.

## 2. Asosiy natijalar

### Bosqich 0 — baseline (eski kod)
Rejadagi bashorat (d=512, N=1000 → 15–25%) **10–25× xato** chiqdi: haqiqiy aniqlik 0.8%.
Sabab: klassik HRR'da d=512 uchun sig'im ~30–50 fakt; N=1000 da signal shovqin ostida.
Element-wise normalizatsiya "bugi" yuqori yuklamada aniqlikni yana ~2–4× tushiradi
(d=1024, N=200: 0.29 → 0.11), lekin past yuklamada deyarli zarar qilmaydi — bu ham
o'lchandi, taxmin qilinmadi.

### Bosqich 1 — FHRR yadro
`bind → unbind → cleanup` identiklik: kosinus 1.000. Unit testlar o'tdi.

### Bosqich 2 — bloklash ishlaydi
d=512, N=1000, round_robin: B=1 → 1.7%, B=8 → 17.8%, B=16 → 53.2%, B=32 → 94.7%.
Monoton o'sish tasdiqlandi. `hash` ≈ `round_robin`; `similarity` kichik B da yomonroq,
B=32 da eng yaxshi (97.1%). Yuklama balansi: rr 1.00–1.03, hash 1.08–1.33.

### Bosqich 3 — 90%+ maqsadga erishildi
**Optimal: d=1024, B=32, N=1000 → aniqlik 1.000**, 1.84 ms/so'rov, ~8.5 MB.
Zaxira variantlar: d=1024/B=16 → 95.1%; d=512/B=32 → 97.1%; d=2048/B=8 → 95.9%.
Kapasitet formulasi `d/(2 ln C)` amaliyotni 1.4–1.8× kam baholaydi (batafsil:
`docs/capacity_formula.md`); amaliy qoida: **blok hajmi ≤ d/10**.

### Bosqich 4 — sparse VSA ishlamadi (halol natija)
d=10000, k=100, thinning-asosli bind/bundle: aniqlik 0.001 (N=1000) — top-k bundle
amalda 1–2 faktni saqlaydi, unbind paytida obyekt signali tasodifiy fillerlar orasida
yo'qoladi. Dense d=2048/B=8 shu nuqtada 95.9% beradi. Xulosa: Frady uslubidagi
to'g'ri kontekstga bog'liq thinning va kattaroq k kerak; hozirgi implementatsiya
ishlatilmaydi. Keyingi qadam sifatida belgilandi.

### Bosqich 5 — resonator
Juftlik xotirasida (har juftlik alohida) 2 iteratsiyada 100% aniqlik (N=100, 500, 1000).
Bitta umumiy bundle ichida esa 0.1–1% — bu kutilgan: superpozitsiyada N juftlik
bir vaqtda saqlanmaydi. 3-argumentli variant 3% (ishlamadi).

### Bosqich 6 — FSRS + LP
Unutish va pruning ishlaydi: 2000 qadamda 941/1000 fakt o'chirildi (R<0.1), shundan
keyin aniqlik 4% → 68% ga ko'tarildi (interferensiya kamaydi). **Muhim salbiy natija:**
R bo'yicha vaznlash aniqlikni tushiradi (0.342 vs 0.556 vaznsiz) — so'ralgan faktning
o'z signali ham w_i ga ko'payadi, shovqin esa faqat RMS bo'yicha tushadi.

## 3. Umumiy xulosa- **"To'lqin ishlaydi"** — 1000 fakt, 100% aniqlik, 1.84 ms/so'rov, 8.5 MB, faqat NumPy.
- Bloklash + noyob kalitlar eng katta yutuq: 1.7% → 100%.
- Sparse VSA va resonator-superpozitsiya hozircha ishlamaydi — bu halol cheklov.
- FSRS vaznlash zararli; unutish/pruning esa foydali.
- Rejadagi bashoratlar (15–25%, sparse 10×, resonator 85%) o'lchovda tasdiqlanmadi;
  faqat 90%+ maqsad haqiqatan erishildi.

## 4. Keyingi qadamlar

1. Sparse VSA: Frady (2018) bo'yicha to'g'ri thinning + k ni oshirish (k=300–500).
2. Resonator: superpozitsiyada N ni kamaytirish yoki blokli+resonator gibrid.
3. FSRS: vaznlashni faqat pruning uchun ishlatish, M ni vaznsiz yig'ish.
4. Ixtiyoriy: mobil prototip (8-bosqich) — hozircha bloklangan.

## 5. Fayllar

```
results/baseline_results.csv          stage0: 6 konfiguratsiya, proper vs buggy
results/stage2_scaling.csv            42 qator: B × strategiya × query-mode
results/stage3_d_sweep.csv            36 konfiguratsiya: d × B
results/stage3_capacity_curve.csv     N90 egri chizig'i
results/stage3_optimal_config.json    {"d":1024,"B":32,"acc":1.0,...}
results/stage4_dense_vs_sparse.csv    dense vs sparse
results/stage5_factorization.csv      resonator natijalari
results/stage6_forgetting_curve.csv   unutish egri chizig'i
results/stage6_weighted_vs_flat.csv   vaznlash taqqoslash
results/final_benchmark.csv           yakuniy jamlama
results/ablation.csv                  komponentlar hissasi
results/scaling_curves.png            grafiklar
```

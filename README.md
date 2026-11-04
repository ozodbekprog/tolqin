# To'lqin — FHRR/VSA xotira tadqiqoti

Fourier HRR (FHRR) asosidagi vektor-simvolik xotira (VSA): faktlar
`bind(bind(S,R), permute(O))` ko'rinishida bitta superpozitsiya xotirasiga
yoziladi va `cleanup` orqali qaytariladi. Repo bloklash, sig'im formulasi,
dense/sparse xotira, resonator faktorizatsiyasi va FSRS unutishini o'lchaydi.

## Tuzilma

```
src/        fhrr.py, factworld.py, blocked_memory.py, baseline_hrr.py,
            resonator.py, sparse_vsa.py, fsrs.py, lp_attention.py,
            integrated_memory.py   # muzlatilgan API (CONTRACT.md)
scripts/    stage0_baseline.py ... stage7_benchmark.py
results/    barcha o'lchov CSV'lari, final_benchmark.csv, ablation.csv,
            scaling_curves.png
docs/       capacity_formula.md  # sig'im formulasi tahlili
tests/      unittest to'plami
```

## Ishga tushirish

Barcha skriptlar repo ildizidan ishlaydi (natijalar `results/` ga yoziladi):

```bash
python3 scripts/stage0_baseline.py        # HRR baseline: proper vs elementwise_buggy
python3 scripts/stage2_scaling.py         # blok masshtablash (B, strategiya, query mode)
python3 scripts/stage3_capacity.py        # d×B sweep + sig'im egri chizig'i
python3 scripts/stage4_dense_vs_sparse.py # dense vs sparse xotira
python3 scripts/stage5_factorization.py   # resonator2/3 vs simple cleanup
python3 scripts/stage6_forgetting.py      # FSRS unutish + weighted vs flat
python3 scripts/stage7_benchmark.py       # yakuniy agregatsiya (qayta o'lchamaydi)
```

`stage7` mavjud CSV'larni o'qib `results/final_benchmark.csv`,
`results/ablation.csv` va `results/scaling_curves.png` ni yaratadi.

## Asosiy natijalar

- **d=1024, B=32 (FHRR):** N=1000 da **100%** aniqlik, 7.64 ms/so'rov,
  8.45 MB. d=1024/B=16 -> 96.0%.
- **Dense d=2048, B=8:** N=1000 da **95.9%**, 4.89 ms/so'rov, 16.5 MB;
  N=5000 da 5.0%, N=10000 da 0.5% gacha tushadi.
- **Sig'im formulasi** `M_cap ≈ d/(2 ln C)`: default dunyoda (50×20 kalit,
  to'qnashuvlar) 4 ta d dan faqat 1 tasi ±20% ichida — **kelmadi**. Kalit
  to'qnashuvisiz dunyoda 3/3 mos keldi, ya'ni formula faqat shovqin bilan
  cheklangan rejimda to'g'ri.
- **Sparse (d=10000, k=100):** juda tez (0.15 ms/so'rov) va kam xotira
  (10 MB), lekin N=1000 da **0.1%**, N≥5000 da **0%** — amaliy cleanup uchun
  yaroqsiz (ma'lum cheklov).
- **Unutish (step 2000):** flat 55.6% vs FSRS weighted 34.2% — vaznlash
  aniqlikni pasaytiradi.
- **Faktorizatsiya:** izolyatsiyalangan juftliklarda resonator2 va simple
  cleanup ikkalasi 100%; birlashtirilgan xotirada resonator2 0.1% (simple
  32.6%) — bundled rejimda resonator ishlamaydi.

## Testlar

```bash
python3 -m unittest discover -s tests -v
```

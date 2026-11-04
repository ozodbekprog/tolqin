# Kapasitet formulasi: nazariya vs amaliyot

**Eksperiment:** `scripts/stage3_capacity.py` → `results/stage3_capacity_curve.csv`.
Dunyo: N ta fakt, har biri noyob `(s, r)` kalitda, obyekt kodkitobi C = N.
B = 1, strategiya ahamiyatsiz. Aniqlik 1000 so'rov bo'yicha (N ≥ 400 da 150 so'rov).

## Nazariya

\[
M_{\text{cap}} \approx \frac{d}{2 \ln C}, \qquad C = 1000 \Rightarrow \ln C \approx 6.91
\]

Interpretatsiya: bitta blokdagi faktlar soni shu chegaradan oshsa, cleanup shovqini
signalni bosib ketadi (signal ~ 1/√M, shovqin ~ √(2 ln C / d)).

## Amaliy o'lchov (N90 — aniqlik 0.90 ga tushadigan N)

| d | Nazariy M_cap | Amaliy N90 | Nisbat (amaliy/nazariy) |
|---|---|---|---|
| 512 | 37 | 66 | 1.78 |
| 1024 | 74 | 125 | 1.69 |
| 2048 | 148 | 227 | 1.53 |
| 4096 | 297 | 419 | 1.41 |

N90 qiymatlari qo'shni nuqtalar orasidagi chiziqli interpolyatsiya bilan olingan.

## Xom egri chiziq

| d | N=50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|
| 512 | 0.960 | 0.770 | 0.355 | 0.120 | 0.024 |
| 1024 | 1.000 | 0.980 | 0.665 | 0.258 | 0.078 |
| 2048 | 1.000 | 1.000 | 0.950 | 0.583 | 0.181 |
| 4096 | 1.000 | 1.000 | 1.000 | 0.920 | 0.508 |

## Xulosa

**Formula mos kelmadi (±20% mezoni bo'yicha), lekin sifat jihatdan to'g'ri.**

- Nisbat 1.41–1.78 oralig'ida: formula kapasitetni **1.4–1.8× kam baholaydi**
  (ya'ni nazariy chegara — pastki bound, amaliyot yaxshiroq).
- Nisbat d oshgani sari 1 ga yaqinlashadi (1.78 → 1.41): katta o'lchamda
  asimptotik rejim yaqinlashadi, kichik d da chekli o'lcham effektlari ustun.
- Amaliy qoida: **blok hajmi M ≤ d/10** dan oshmasin — bu 90%+ aniqlikni
  deyarli kafolatlaydi (barcha o'lchangan d larda N90 > d/10).

## Nima uchun muhim

Stage 2/3 da bloklash strategiyasi tanlashda B ni shu qoida bo'yicha tanladik:
`B ≥ N / (d/10)`. N=1000, d=1024 → B ≥ 9.8 → B ∈ {16, 32} ishlatildi va
90%+ berdi (`stage3_optimal_config.json`).

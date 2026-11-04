# VSA vs hash — adolatli taqqoslash (stage 8)

**Skript:** `scripts/stage8_baselines.py` → `results/stage8_baselines.csv`
**Dunyo:** 200 subject × 5 object = 1000 juftlik, C = 1000 object, d = 1024.
**VSA:** juftlik kodlash `f = bind(S, permute(O,1))`, bloklash `hash(subject)` bo'yicha
(bitta subjectning hamma objecti bitta blokda). Metrika: 5 ta bog'langan objectdan
nechtasi top-5 ichida (recall/precision).

## 1. Qisman kalit (faqat subject berilgan)

| Usul | Recall@5 | Precision@5 | ms/so'rov | Xotira |
|---|---|---|---|---|
| Inverted index (s → [o...]) | 1.000 | 1.000 | 0.0010 | 0.08 MB |
| Hash KV (faqat to'liq kalit) | 0.000 | 0.000 | — | 0.14 MB |
| VSA, B=1 | 0.123 | 0.123 | 0.10 | 8.20 MB |
| VSA, B=8 | 0.703 | 0.703 | 0.53 | 8.26 MB |
| **VSA, B=32** | **0.988** | **0.988** | 1.97 | 8.45 MB |

## 2. Shovqinli kalit

`level` = har bir komponentga qo'shilgan kompleks shovqin amplitudasi (FHRR
komponentlari |v_j| = 1). `cue_sim` — shovqinli kalitning asl kalit bilan kosinusi.

| Shovqin | cue_sim | Exact lookup | VSA B=8 | VSA B=32 |
|---|---|---|---|---|
| 10% | 0.995 | 0.000 | 0.980 | **1.000** |
| 30% | 0.958 | 0.000 | 0.975 | **1.000** |
| 50% | 0.894 | 0.000 | 0.930 | **1.000** |
| 100% | 0.707 | 0.000 | 0.605 | **1.000** |
| 200% | 0.447 | 0.000 | 0.125 | 0.500 |

## 3. Xulosa — "kill test" o'tdi, lekin shart bilan

**O'tdi:** qisman kalit 98.8%, 30% shovqinda 100%, 100% shovqinda ham 100% (B=32).
Rejadagi qizil chiziqlar: qisman kalit <30% → pivot (bizda 98.8%, o'tdi);
10% shovqinda <40% → pivot (bizda 100%, o'tdi).

**Shart:** toza (shovqinsiz) qisman kalit uchun oddiy **inverted index** ham 100%
beradi va u **~2000× tez, ~100× kam xotira** oladi. Ya'ni VSA faqat quyidagi
holatlarda oqlanadi:

1. kalit **shovqinli / buzilgan** (hash umuman ishlamaydi — 0%);
2. **o'xshashlik bo'yicha** qidirish kerak (eng yaqin fakt, kesh, assotsiatsiya);
3. kalitlar **kompozitsion** (bind orqali yasalgan, alohida saqlanmaydi).

Toza, aniq kalitlar uchun VSA — qimmat hash table. Bu rejadagi taxmin **tasdiqlandi**.

## 4. Keyingi qadam

- FSRS vaznlashni olib tashlash, faqat pruning qoldirish (reja #4);
- N=10000 scaling (reja #6) — dense B=32 bilan;
- Sparse VSA (reja #5) — Frady algoritmi, 5 kundan keyin <30% bo'lsa to'xtatish.

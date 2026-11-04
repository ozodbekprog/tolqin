"""Stage 0 — klassik real qiymatli HRR (circular convolution) baseline.

Bu "to'lqin vs transformer" eski kodining qayta tiklangan varianti: FHRR
(frozen `src/fhrr.py`) emas, balki Plate'ning klassik HRR'i — real Gauss
vektorlar + FFT orqali circular convolution.

Muhim: `cleanup_elementwise_buggy` — eski koddagi xato. U vektorni global L2
norma bilan emas, har bir komponentni alohida normallaydi (r / |r| elementwise).
Bu signal va shovqin amplituda farqini yo'qotadi va faqat ishora (sign)
ma'lumotini qoldiradi.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float64


# --------------------------------------------------------------------------
# Asosiy operatsiyalar
# --------------------------------------------------------------------------
def random_hrr(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Real Gauss vektori, unit L2 norma."""
    rng = np.random.default_rng() if rng is None else rng
    v = rng.standard_normal(d)
    return (v / np.linalg.norm(v)).astype(DTYPE)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution: irfft(rfft(a) * rfft(b))."""
    d = a.shape[0]
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=d).astype(DTYPE)


def unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular correlation: irfft(rfft(c) * conj(rfft(b)))."""
    d = c.shape[0]
    return np.fft.irfft(np.fft.rfft(c) * np.conj(np.fft.rfft(b)), n=d).astype(DTYPE)


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(v, k)


def inv_permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(v, -k)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------
def build_obj_matrix(codebook: dict) -> tuple[list[str], np.ndarray]:
    """(C,d) matritsa, qatorlar L2-normallashgan (cosine uchun)."""
    names = list(codebook.keys())
    mat = np.stack([np.asarray(codebook[n], dtype=DTYPE) for n in names])
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return names, (mat / norms).astype(DTYPE)


def cleanup_proper_matrix(r: np.ndarray, names: list[str], mat_n: np.ndarray) -> tuple[str, float]:
    """To'g'ri cleanup: r ni L2 normallab, cosine bo'yicha eng yaxshisi."""
    rn = r / (np.linalg.norm(r) + 1e-12)
    sims = mat_n @ rn
    i = int(np.argmax(sims))
    return names[i], float(sims[i])


def cleanup_proper(r: np.ndarray, codebook: dict) -> tuple[str, float]:
    names, mat_n = build_obj_matrix(codebook)
    return cleanup_proper_matrix(r, names, mat_n)


def cleanup_buggy_matrix(r: np.ndarray, names: list[str], mat_n: np.ndarray,
                         mat_q: np.ndarray | None = None) -> tuple[str, float]:
    """ESKI XATO: elementwise normalizatsiya (r / (|r|+1e-12)), keyin dot/d.

    Bu vektorning global amplitudasini (signal vs. shovqin energiyasi) tashlab
    yuboradi — har bir komponent faqat ishorasini saqlaydi. Natijada kodkitob
    vektorlari bilan solishtirish "1-bitli" bo'lib qoladi va retrieved vektorda
    signal shovqin ichida bo'lsa farqlash qobiliyati yo'qoladi.

    mat_q: oldindan kvantlangan kodkitob (bir marta hisoblash uchun); vaqt
    o'lchovida xatoning o'zi emas, faqat cleanup ishi hisobga olinadi.
    """
    d = r.shape[0]
    rq = r / (np.abs(r) + 1e-12)          # elementwise, global norma EMAS
    if mat_q is None:
        mat_q = mat_n / (np.abs(mat_n) + 1e-12)
    sims = (mat_q @ rq) / d
    i = int(np.argmax(sims))
    return names[i], float(sims[i])


def quantize_elementwise(mat: np.ndarray) -> np.ndarray:
    """Kodkitobni elementwise normallash (xato varianti uchun, bir marta)."""
    return (mat / (np.abs(mat) + 1e-12)).astype(DTYPE)


def cleanup_elementwise_buggy(r: np.ndarray, codebook: dict) -> tuple[str, float]:
    names, mat_n = build_obj_matrix(codebook)
    return cleanup_buggy_matrix(r, names, mat_n)


# --------------------------------------------------------------------------
# Fakt dunyosi (src/factworld.py ga analog)
# --------------------------------------------------------------------------
def make_world(n_facts: int = 1000, d: int = 512, n_subjects: int = 50,
               n_relations: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    subjects = {f"s{i}": random_hrr(d, rng) for i in range(n_subjects)}
    relations = {f"r{i}": random_hrr(d, rng) for i in range(n_relations)}
    objects = {f"o{i}": random_hrr(d, rng) for i in range(n_facts)}  # C = N
    facts = []
    for i in range(n_facts):
        s = f"s{int(rng.integers(n_subjects))}"
        r = f"r{int(rng.integers(n_relations))}"
        facts.append((s, r, f"o{i}"))
    return {"subjects": subjects, "relations": relations, "objects": objects,
            "facts": facts, "d": d}


def encode_fact(fw: dict, s: str, r: str, o: str, perm: int = 1) -> np.ndarray:
    return bind(bind(fw["subjects"][s], fw["relations"][r]),
                permute(fw["objects"][o], perm))


def query_key(fw: dict, s: str, r: str) -> np.ndarray:
    return bind(fw["subjects"][s], fw["relations"][r])


def retrieve(fw: dict, M: np.ndarray, s: str, r: str, perm: int = 1) -> np.ndarray:
    return inv_permute(unbind(unbind(M, fw["subjects"][s]), fw["relations"][r]), perm)


def bundle(vectors) -> np.ndarray:
    return np.sum(np.stack(list(vectors)), axis=0).astype(DTYPE)


def accuracy_matrix(fw: dict, M: np.ndarray, names: list[str], mat_n: np.ndarray,
                    cleanup_fn, perm: int = 1) -> tuple[float, float]:
    """(accuracy, elapsed_ms) — barcha faktlar bo'yicha cleanup natijasi."""
    import time
    t0 = time.perf_counter()
    correct = 0
    for s, r, o in fw["facts"]:
        cand = retrieve(fw, M, s, r, perm)
        name, _ = cleanup_fn(cand, names, mat_n)
        correct += int(name == o)
    ms = (time.perf_counter() - t0) * 1e3
    return correct / len(fw["facts"]), ms

"""FHRR yadro — unit-magnitude kompleks vektorlar (Fourier HRR).

MUZLATILGAN: faqat asosiy agent o'zgartiradi.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.complex64


def random_fhrr(d: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=d)).astype(DTYPE)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).astype(DTYPE)


def unbind(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (c * np.conj(b)).astype(DTYPE)


def permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(v, k)


def inv_permute(v: np.ndarray, k: int = 1) -> np.ndarray:
    return np.roll(v, -k)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.real(np.vdot(a, b)) / denom)


def build_obj_matrix(codebook: dict) -> tuple[list[str], np.ndarray]:
    """Kodkitobni (C,d) normallashgan matritsaga aylantiradi (bir marta)."""
    names = list(codebook.keys())
    mat = np.stack([codebook[n] for n in names]).astype(DTYPE)
    mat_n = (mat / np.sqrt(mat.shape[1])).astype(DTYPE)  # unit-magnitude -> row norm 1
    return names, mat_n


def cleanup_matrix(r: np.ndarray, names: list[str], mat_n: np.ndarray) -> tuple[str, float, np.ndarray]:
    rn = r / (np.linalg.norm(r) + 1e-12)
    sims = np.real(mat_n @ np.conj(rn))
    i = int(np.argmax(sims))
    return names[i], float(sims[i]), sims


def cleanup_proper(r: np.ndarray, codebook: dict) -> tuple[str, float]:
    names, mat_n = build_obj_matrix(codebook)
    name, sim, _ = cleanup_matrix(r, names, mat_n)
    return name, sim

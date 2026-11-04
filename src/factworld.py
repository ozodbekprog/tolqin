"""Sintetik fakt dunyosi: (subject, relation, object) uchliklari.

MUZLATILGAN: faqat asosiy agent o'zgartiradi.
"""
from __future__ import annotations

import numpy as np

from src.fhrr import DTYPE, bind, build_obj_matrix, cleanup_matrix, inv_permute, permute, random_fhrr, unbind


def make_world(n_facts: int = 1000, d: int = 512, n_subjects: int = 50,
               n_relations: int = 20, seed: int = 0, unique_keys: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    subjects = {f"s{i}": random_fhrr(d, rng) for i in range(n_subjects)}
    relations = {f"r{i}": random_fhrr(d, rng) for i in range(n_relations)}
    objects = {f"o{i}": random_fhrr(d, rng) for i in range(n_facts)}
    facts = []
    if unique_keys and n_facts <= n_subjects * n_relations:
        # (s, r) kalitlari takrorlanmaydi: aks holda bitta kalitga bir nechta obyekt
        # birikadi va cleanup strukturaviy shiftga uriladi (o'lchangan: 0.63 @ N=1000).
        idx = rng.choice(n_subjects * n_relations, size=n_facts, replace=False)
        for k, i in enumerate(idx):
            facts.append((f"s{int(i) // n_relations}", f"r{int(i) % n_relations}", f"o{k}"))
    else:
        for i in range(n_facts):
            s = f"s{int(rng.integers(n_subjects))}"
            r = f"r{int(rng.integers(n_relations))}"
            facts.append((s, r, f"o{i}"))
    return {"subjects": subjects, "relations": relations, "objects": objects,
            "facts": facts, "d": d}


def encode_fact(fw: dict, s: str, r: str, o: str, perm: int = 1) -> np.ndarray:
    return bind(bind(fw["subjects"][s], fw["relations"][r]), permute(fw["objects"][o], perm))


def query_key(fw: dict, s: str, r: str) -> np.ndarray:
    return bind(fw["subjects"][s], fw["relations"][r])


def retrieve(fw: dict, M: np.ndarray, s: str, r: str, perm: int = 1) -> np.ndarray:
    return inv_permute(unbind(unbind(M, fw["subjects"][s]), fw["relations"][r]), perm)


def bundle(vectors) -> np.ndarray:
    M = np.sum(np.stack(list(vectors)), axis=0)
    return M.astype(DTYPE)


def query_all(fw: dict, M: np.ndarray, perm: int = 1) -> list[tuple[str, float]]:
    names, mat_n = build_obj_matrix(fw["objects"])
    out = []
    for s, r, _ in fw["facts"]:
        cand = retrieve(fw, M, s, r, perm)
        name, sim, _ = cleanup_matrix(cand, names, mat_n)
        out.append((name, sim))
    return out


def accuracy(fw: dict, M: np.ndarray, perm: int = 1) -> float:
    preds = query_all(fw, M, perm)
    correct = sum(1 for (s, r, o), (name, _) in zip(fw["facts"], preds) if name == o)
    return correct / len(fw["facts"])

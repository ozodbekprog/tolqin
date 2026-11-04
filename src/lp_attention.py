"""LP layer: labile precision / attention over a set of facts.

An item is "salient" exactly when its retention is about to change, so the
attention weight is proportional to the predicted retention drop over the next
``dt``:

    att_i = |R_i(t + dt) - R_i(t)| / sum_j |R_j(t + dt) - R_j(t)|

The weights sum to 1. If no item changes at all (e.g. every item was just
reviewed, or the set is empty) the weights fall back to a uniform distribution
(empty dict for an empty set); this is documented rather than raising so that
callers can sample safely.
"""
from __future__ import annotations

from .fsrs import DEFAULT_A, DEFAULT_B, FSRS


class LPAttention:
    """Per-fact FSRS instances plus the derived attention distribution."""

    def __init__(self, a: float = DEFAULT_A, b: float = DEFAULT_B,
                 S0: float = 1.0):
        self.a = float(a)
        self.b = float(b)
        self.S0 = float(S0)
        self.fsrs: dict = {}   # id -> FSRS

    # -- lifecycle ---------------------------------------------------------
    def add_fact(self, i, t0: float = 0.0, S0: float | None = None) -> FSRS:
        if i in self.fsrs:
            raise KeyError(f"fact {i!r} already tracked")
        self.fsrs[i] = FSRS(tau=t0, S=self.S0 if S0 is None else S0,
                            a=self.a, b=self.b)
        return self.fsrs[i]

    def drop_fact(self, i) -> None:
        self.fsrs.pop(i, None)

    def __len__(self) -> int:
        return len(self.fsrs)

    def __contains__(self, i) -> bool:
        return i in self.fsrs

    # -- dynamics ----------------------------------------------------------
    def R(self, i, t: float) -> float:
        return self.fsrs[i].R(t)

    def recall(self, i, t: float, success: bool = True) -> float:
        return self.fsrs[i].on_recall(t, success)

    # -- attention ---------------------------------------------------------
    def weights(self, t: float, dt: float = 1.0) -> dict:
        if not self.fsrs:
            return {}
        drops = {i: abs(f.R(t + dt) - f.R(t)) for i, f in self.fsrs.items()}
        total = sum(drops.values())
        if total <= 0.0:                       # nothing changing -> uniform
            u = 1.0 / len(drops)
            return {i: u for i in drops}
        return {i: d / total for i, d in drops.items()}

    def active(self, t: float, threshold: float = 0.5) -> list:
        """Ids whose retention is above ``threshold`` (surviving memories)."""
        return [i for i, f in self.fsrs.items() if f.R(t) > threshold]

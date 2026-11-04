"""FSRS-lite forgetting model (power-law style retention).

Retention of a memory item reviewed at time ``tau`` with stability ``S``:

    R(t) = 1 / (1 + max(t - tau, 0) / (9 * S))

so ``R(tau) = 1`` and ``R(tau + 9*S) = 0.5`` (the half-life is ``9*S``).
A successful recall grows stability in proportion to the *forgetting* that
happened since the last review, ``S *= 1 + a*(1 - R_now)``: reviewing a nearly
forgotten item is worth much more than re-reading a fresh one. A failed recall
shrinks stability geometrically, ``S *= exp(-b)``. In both cases ``tau`` is
reset to the review time.

Defaults ``a = 0.5`` (max +50% stability per successful review) and
``b = 0.2`` (about -18% stability per failed review). The default ``S = 1``
means "one unit of stability", i.e. a half-life of 9 time units; callers that
simulate longer horizons should pass a larger ``S``.
"""
from __future__ import annotations

import math

DEFAULT_A = 0.5
DEFAULT_B = 0.2
HALF_LIFE_K = 9.0


class FSRS:
    """Single-item retention model. See module docstring for the equations."""

    def __init__(self, tau: float = 0.0, S: float = 1.0,
                 a: float = DEFAULT_A, b: float = DEFAULT_B):
        if S <= 0:
            raise ValueError("stability S must be positive")
        self.tau = float(tau)   # last review time
        self.S = float(S)       # stability (half-life = 9*S)
        self.a = float(a)
        self.b = float(b)

    def R(self, t: float) -> float:
        """Retention at time t in [0, 1]."""
        dt = max(float(t) - self.tau, 0.0)
        return 1.0 / (1.0 + dt / (HALF_LIFE_K * self.S))

    def on_recall(self, t: float, success: bool) -> float:
        """Apply a review at time t; returns retention just before the review."""
        r_now = self.R(t)
        if success:
            self.S *= 1.0 + self.a * (1.0 - r_now)
        else:
            self.S *= math.exp(-self.b)
        self.tau = float(t)
        return r_now

    def __repr__(self) -> str:
        return f"FSRS(tau={self.tau:g}, S={self.S:g}, a={self.a:g}, b={self.b:g})"

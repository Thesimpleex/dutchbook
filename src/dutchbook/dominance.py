"""Accuracy-dominance certificates (Joyce 1998; Predd et al. 2009).

Let ``D`` be the Bregman divergence of an additive strictly proper scoring rule and
``p*`` the projection of ``q`` onto the convex coherent set ``C`` that minimises
``D(p, q)`` over ``p ∈ C``. The generalised Pythagorean inequality says that for every
``v ∈ C``, in particular every world ``v_w``,

    D(v, q) ≥ D(v, p*) + D(p*, q).

Since ``D(v_w, y)`` is the score ``S(y, w)`` of forecast ``y`` in world ``w``, this reads

    S(q, w) - S(p*, w) ≥ D(p*, q)    for every world w,

so the projected vector strictly improves the score in *every* world whenever ``q`` is
incoherent. The statement concerns the whole forecast vector, not any sub-block of it.

With an approximate minimiser ``p`` the three-point identity gives the exact margin
``S(q, w) - S(p, w) - D(p, q) = <∇_p D(p, q), v_w - p>``, which is at least ``-gap`` for the
Frank-Wolfe gap of ``p``; the certificate therefore tests ``margin ≥ -gap - tol``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import xlog1py, xlogy

from dutchbook.projection import kl_divergence

Scoring = Literal["brier", "log"]


def brier_scores(W: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``Σ_j (q_j - W_wj)²`` for every world ``w`` (sum, not mean, over formulas)."""
    return ((np.asarray(W, dtype=float) - np.asarray(q, dtype=float)) ** 2).sum(axis=1)


def log_scores(W: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Log loss ``-Σ_j [W_wj log q_j + (1 - W_wj) log(1 - q_j)]`` for every world."""
    W = np.asarray(W, dtype=float)
    q = np.asarray(q, dtype=float)
    return -(xlogy(W, q) + xlog1py(1.0 - W, -q)).sum(axis=1)


@dataclass(frozen=True)
class DominanceCertificate:
    """Per-world accuracy comparison of a forecast ``q`` and its projection ``p``.

    Attributes
    ----------
    scoring : str
        ``"brier"`` (sum of squared errors) or ``"log"`` (log loss).
    divergence : float
        ``D(p, q)``: ``‖p - q‖²`` for Brier, ``Σ_j KL(p_j ‖ q_j)`` for log.
    score_before, score_after : ndarray
        Score of ``q`` and ``p`` in every distinct world.
    improvement : ndarray
        ``score_before - score_after`` per world.
    margin : ndarray
        ``improvement - divergence``; non-negative at the exact projection.
    gap : float
        Frank-Wolfe duality gap of ``p``.
    holds : bool
        ``min(margin) ≥ -gap - tol``: the generalised Pythagorean inequality holds.
    """

    scoring: str
    divergence: float
    score_before: np.ndarray
    score_after: np.ndarray
    improvement: np.ndarray
    margin: np.ndarray
    gap: float
    holds: bool

    @property
    def min_improvement(self) -> float:
        return float(self.improvement.min())

    @property
    def strictly_dominates(self) -> bool:
        """Every world strictly prefers ``p`` to ``q``."""
        return self.min_improvement > 0.0


def dominance_certificate(
    W: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    scoring: Scoring = "brier",
    gap: float = 0.0,
    tol: float = 1e-9,
) -> DominanceCertificate:
    """Check ``S(q, w) - S(p, w) ≥ D(p, q)`` in every world ``w`` (rows of ``W``).

    For ``scoring="log"`` pass the clipped ``q`` returned by the projection, since the
    log score of an exact 0 or 1 is infinite.
    """
    W = np.asarray(W, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.asarray(p, dtype=float)
    if scoring == "brier":
        before, after = brier_scores(W, q), brier_scores(W, p)
        divergence = float(np.sum((p - q) ** 2))
    elif scoring == "log":
        before, after = log_scores(W, q), log_scores(W, p)
        divergence = kl_divergence(p, q)
    else:
        raise ValueError("scoring must be 'brier' or 'log'")
    improvement = before - after
    margin = improvement - divergence
    return DominanceCertificate(
        scoring=scoring,
        divergence=divergence,
        score_before=before,
        score_after=after,
        improvement=improvement,
        margin=margin,
        gap=float(gap),
        holds=bool(margin.min() >= -gap - tol),
    )

"""Coherence of forecasts and Dutch-book certificates via linear programming.

Setting (de Finetti 1937). A forecaster quotes a price ``q_j`` for a contract that pays
1 if formula ``φ_j`` is true. A bettor chooses stakes ``s_j`` (positive: buy the
contract at ``q_j``, negative: sell it) and earns ``Σ_j s_j (W_wj - q_j)`` in world ``w``,
where ``W`` is the 0/1 truth matrix. The forecaster is coherent iff no stake vector
guarantees a strictly positive payoff, iff ``q`` lies in the convex hull of the rows of
``W`` (the coherent polytope ``C``).

Duality. For a norm on the stakes with dual norm ``‖·‖_*``,

    max_{‖s‖ ≤ 1} min_w s·(W_w - q)  =  min_{λ ∈ Δ} ‖Wᵀλ - q‖_*  =  dist_*(q, C).

Total stake ``‖s‖₁ ≤ 1`` therefore prices the Dutch book at the sup-norm distance to
``C`` (Andrews and Sarkar 2026); ``‖s‖_∞ ≤ 1`` at the ℓ1 distance; ``‖s‖₂ ≤ 1`` at the
Euclidean distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from dutchbook.projection import euclidean_projection

StakeNorm = Literal["l1", "linf", "l2"]
STAKE_NORMS: tuple[str, ...] = ("l1", "linf", "l2")
_DUAL_NORM = {"l1": "linf", "linf": "l1", "l2": "l2"}

DEFAULT_TOL = 1e-9
_HIGHS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


def check_matrix(W: np.ndarray) -> np.ndarray:
    """Validate a truth matrix of shape ``(worlds, formulas)`` with entries in {0, 1}."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("W must be a non-empty 2-D truth matrix (worlds x formulas)")
    if not np.isin(W, (0.0, 1.0)).all():
        raise ValueError("W must contain only 0 and 1")
    return W


def check_forecasts(q: Sequence[float] | np.ndarray, m: int) -> np.ndarray:
    """Validate a forecast vector: finite, in [0, 1], one entry per formula."""
    q = np.asarray(q, dtype=float)
    if q.shape != (m,):
        raise ValueError(f"expected {m} forecasts, got shape {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("forecasts must be finite")
    if (q < 0).any() or (q > 1).any():
        raise ValueError("forecasts must lie in [0, 1]")
    return q


@dataclass(frozen=True)
class Distance:
    """Distance from ``q`` to the coherent polytope.

    Attributes
    ----------
    value : float
        ``min_λ ‖Wᵀλ - q‖`` in the requested norm.
    norm : str
        ``"linf"``, ``"l1"`` or ``"l2"``.
    weights : ndarray
        Optimal distribution ``λ`` over the rows of ``W`` (a probability measure on worlds).
    point : ndarray
        ``Wᵀλ``, the closest coherent forecast in that norm.
    """

    value: float
    norm: str
    weights: np.ndarray
    point: np.ndarray


def coherence_distance(
    W: np.ndarray, q: Sequence[float] | np.ndarray, norm: Literal["linf", "l1", "l2"] = "linf"
) -> Distance:
    """Distance from ``q`` to the convex hull of the rows of ``W`` (exact LP / QP).

    ``norm="linf"`` and ``"l1"`` are solved as linear programs with HiGHS; ``"l2"`` with
    the Wolfe min-norm-point algorithm (:func:`dutchbook.projection.euclidean_projection`).
    """
    W = check_matrix(W)
    k, m = W.shape
    q = check_forecasts(q, m)
    if norm == "l2":
        proj = euclidean_projection(W, q)
        return Distance(
            float(np.linalg.norm(proj.point - q)), "l2", proj.dense_weights(W), proj.point
        )
    if norm not in ("linf", "l1"):
        raise ValueError("norm must be 'linf', 'l1' or 'l2'")
    Wt = sparse.csr_matrix(W.T)
    ones_k = sparse.csr_matrix(np.ones((1, k)))
    if norm == "linf":
        # variables [λ (k), t]; min t  s.t.  ±(Wᵀλ - q) ≤ t, Σλ = 1
        t_col = sparse.csr_matrix(np.ones((m, 1)))
        a_ub = sparse.vstack(
            [sparse.hstack([Wt, -t_col]), sparse.hstack([-Wt, -t_col])], format="csr"
        )
        b_ub = np.concatenate([q, -q])
        a_eq = sparse.hstack([ones_k, sparse.csr_matrix((1, 1))], format="csr")
        c = np.zeros(k + 1)
        c[-1] = 1.0
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=[1.0],
            bounds=[(0, None)] * (k + 1),
            method="highs",
            options=_HIGHS,
        )
    else:
        # variables [λ (k), e⁺ (m), e⁻ (m)]; min Σ(e⁺+e⁻)  s.t.  Wᵀλ - e⁺ + e⁻ = q, Σλ = 1
        eye = sparse.identity(m, format="csr")
        a_eq = sparse.vstack(
            [
                sparse.hstack([Wt, -eye, eye]),
                sparse.hstack([ones_k, sparse.csr_matrix((1, 2 * m))]),
            ],
            format="csr",
        )
        c = np.concatenate([np.zeros(k), np.ones(2 * m)])
        res = linprog(
            c,
            A_eq=a_eq,
            b_eq=np.concatenate([q, [1.0]]),
            bounds=[(0, None)] * (k + 2 * m),
            method="highs",
            options=_HIGHS,
        )
    if not res.success:
        raise RuntimeError(f"distance LP failed: {res.message}")
    weights = np.clip(res.x[:k], 0.0, None)
    weights /= weights.sum()
    return Distance(float(res.fun), norm, weights, W.T @ weights)


def is_coherent(W: np.ndarray, q: Sequence[float] | np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """Whether ``q`` is a convex combination of the rows of ``W`` (up to ``tol``).

    Equivalent to feasibility of ``Wᵀλ = q, λ ∈ Δ``; evaluated through the sup-norm
    distance, which is a well-conditioned LP and yields a certificate either way.
    """
    return coherence_distance(W, q, "linf").value <= tol


@dataclass(frozen=True)
class Order:
    """One bet of a Dutch book, placed at the forecaster's own price."""

    index: int
    label: str
    side: Literal["BUY", "SELL"]
    price: float
    stake: float

    @property
    def cost(self) -> float:
        """Cash the bettor pays now (negative: receives)."""
        return self.stake * self.price if self.side == "BUY" else -self.stake * self.price


@dataclass(frozen=True)
class DutchBook:
    """Result of :func:`find_dutch_book`.

    Attributes
    ----------
    stakes : ndarray
        Signed stakes per formula (positive: buy from the forecaster, negative: sell).
        All zero when the forecasts are coherent.
    payoffs : ndarray
        Bettor's net payoff in every row (distinct world) of ``W``.
    profit : float
        ``min(payoffs)``: the payoff guaranteed in every world, per unit of stake norm.
    stake_norm : str
        The norm bounding the stakes (``‖stakes‖ ≤ 1``).
    weights : ndarray
        Optimal dual: the coherent mixture over worlds closest to ``q`` in the dual norm.
    coherent : bool
        True when no Dutch book exists (``profit ≤ tol``).
    """

    stakes: np.ndarray
    payoffs: np.ndarray
    profit: float
    stake_norm: str
    weights: np.ndarray
    coherent: bool

    def orders(
        self,
        q: Sequence[float] | np.ndarray,
        labels: Sequence[str] | None = None,
        scale: float = 1.0,
        min_stake: float = 1e-9,
    ) -> list[Order]:
        """Explicit buy and sell orders, optionally scaled to a stake budget.

        ``scale`` multiplies all stakes (and hence every payoff) by a constant, for
        example the total amount one is prepared to wager under the ℓ1 norm.
        """
        q = np.asarray(q, dtype=float)
        names = list(labels) if labels is not None else [f"#{j}" for j in range(len(q))]
        out = []
        for j, s in enumerate(self.stakes * scale):
            if abs(s) > min_stake * max(scale, 1.0):
                out.append(
                    Order(j, names[j], "BUY" if s > 0 else "SELL", float(q[j]), abs(float(s)))
                )
        return out


def find_dutch_book(
    W: np.ndarray,
    q: Sequence[float] | np.ndarray,
    stake_norm: StakeNorm = "l1",
    tol: float = DEFAULT_TOL,
) -> DutchBook:
    """Maximise the guaranteed profit of a bettor against the forecasts ``q``.

    Solves ``max_s min_w Σ_j s_j (W_wj - q_j)`` subject to ``‖s‖ ≤ 1`` for the chosen
    stake norm. For ``"l1"`` and ``"linf"`` this is one LP over the distinct worlds; for
    ``"l2"`` the optimal stakes are the normalised projection residual ``(p* - q)/‖p* - q‖``
    (Euclidean projection ``p*`` onto the coherent polytope), which Pythagoras certifies:
    ``(p* - q)·(W_w - q) ≥ ‖p* - q‖²`` for every world.

    By strong duality ``profit`` equals the distance from ``q`` to the coherent polytope
    in the dual norm; the LP multipliers on the world constraints are returned as
    ``weights`` (the closest coherent mixture).
    """
    W = check_matrix(W)
    k, m = W.shape
    q = check_forecasts(q, m)
    if stake_norm not in STAKE_NORMS:
        raise ValueError(f"stake_norm must be one of {STAKE_NORMS}")
    if stake_norm == "l2":
        proj = euclidean_projection(W, q)
        resid = proj.point - q
        dist = float(np.linalg.norm(resid))
        weights = proj.dense_weights(W)
        if dist <= tol:
            return _no_book(W, "l2", weights)
        return _finish(W, q, resid / dist, "l2", weights, tol)

    dev = W - q  # (k, m): payoff per unit stake in each world
    if stake_norm == "l1":
        # variables [u (m), v (m), t] with s = u - v; max t s.t. t ≤ dev (u - v), Σ(u+v) ≤ 1
        a_worlds = np.hstack([-dev, dev, np.ones((k, 1))])
        a_norm = np.concatenate([np.ones(2 * m), [0.0]])[None, :]
        a_ub = sparse.csr_matrix(np.vstack([a_worlds, a_norm]))
        b_ub = np.concatenate([np.zeros(k), [1.0]])
        bounds = [(0, None)] * (2 * m) + [(None, None)]
        c = np.zeros(2 * m + 1)
    else:
        a_ub = sparse.csr_matrix(np.hstack([-dev, np.ones((k, 1))]))
        b_ub = np.zeros(k)
        bounds = [(-1, 1)] * m + [(None, None)]
        c = np.zeros(m + 1)
    c[-1] = -1.0
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=_HIGHS)
    if not res.success:
        raise RuntimeError(f"Dutch-book LP failed: {res.message}")
    stakes = res.x[:m] - res.x[m : 2 * m] if stake_norm == "l1" else res.x[:m]
    weights = np.clip(-res.ineqlin.marginals[:k], 0.0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(k, 1.0 / k)
    return _finish(W, q, stakes, stake_norm, weights, tol)


def _no_book(W: np.ndarray, stake_norm: str, weights: np.ndarray) -> DutchBook:
    k, m = W.shape
    return DutchBook(np.zeros(m), np.zeros(k), 0.0, stake_norm, weights, True)


def _finish(
    W: np.ndarray,
    q: np.ndarray,
    stakes: np.ndarray,
    stake_norm: str,
    weights: np.ndarray,
    tol: float,
) -> DutchBook:
    payoffs = (W - q) @ stakes
    profit = float(payoffs.min())
    if profit <= tol:
        return _no_book(W, stake_norm, weights)
    stakes = np.where(np.abs(stakes) < 1e-12, 0.0, stakes)
    return DutchBook(stakes, (W - q) @ stakes, profit, stake_norm, weights, False)


def stake_norm_value(stakes: np.ndarray, stake_norm: str) -> float:
    """Norm of a stake vector in the named stake norm."""
    if stake_norm == "l1":
        return float(np.abs(stakes).sum())
    if stake_norm == "linf":
        return float(np.abs(stakes).max())
    return float(np.linalg.norm(stakes))


def dual_distance_norm(stake_norm: str) -> str:
    """Distance norm that equals the profit under a given stake norm."""
    return _DUAL_NORM[stake_norm]

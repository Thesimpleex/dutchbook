"""Bregman projections onto the coherent polytope.

The coherent polytope is ``C = conv{v_w}`` where ``v_w`` is the 0/1 vector of truth values
of the formulas in world ``w``. Both projections only need a *linear minimisation
oracle*: given a direction ``g`` return a vertex minimising ``<g, v>``. With enumerated
worlds the oracle is a matrix-vector product; ``dutchbook.oracle`` provides a MILP
oracle for formula sets whose worlds cannot be enumerated.

* :func:`euclidean_projection` minimises ``‖p - q‖²`` (the Brier divergence) with Wolfe's
  min-norm-point algorithm (Wolfe 1976), which terminates finitely and stops on the
  Frank-Wolfe duality gap.
* :func:`kl_projection` minimises ``Σ_j KL(p_j ‖ q_j)`` (the log-score divergence) with
  away-step Frank-Wolfe and exact line search, again stopping on the duality gap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import brentq
from scipy.special import rel_entr

from dutchbook.logic import pattern_keys

__all__ = [
    "EnumeratedVertices",
    "Projection",
    "VertexOracle",
    "euclidean_projection",
    "kl_divergence",
    "kl_projection",
]


class VertexOracle(Protocol):
    """Linear minimisation oracle: a vertex of the polytope minimising ``<direction, v>``."""

    def __call__(self, direction: np.ndarray) -> np.ndarray: ...


class EnumeratedVertices:
    """Oracle over explicitly enumerated vertices (rows of ``W``)."""

    def __init__(self, W: np.ndarray) -> None:
        self.W = np.asarray(W, dtype=float)

    def __call__(self, direction: np.ndarray) -> np.ndarray:
        return self.W[int(np.argmin(self.W @ direction))]


@dataclass(frozen=True)
class Projection:
    """Result of a Bregman projection.

    Attributes
    ----------
    point : ndarray
        The projected forecast ``p``, a convex combination of ``vertices``.
    q : ndarray
        The forecast actually projected (for the KL projection ``q`` is clipped to
        ``[eps, 1 - eps]``).
    vertices, weights : ndarray
        Active vertices (worlds) and the convex weights realising ``point``.
    gap : float
        Frank-Wolfe duality gap at ``point``. It bounds the suboptimality of the
        divergence and the amount by which the per-world dominance margin can fall below
        the divergence (see :mod:`dutchbook.dominance`).
    iterations : int
        Number of oracle calls.
    """

    point: np.ndarray
    q: np.ndarray
    vertices: np.ndarray
    weights: np.ndarray
    gap: float
    iterations: int

    def dense_weights(self, W: np.ndarray) -> np.ndarray:
        """Weights over all rows of ``W`` (rows not in the active set get zero)."""
        W = np.asarray(W, dtype=float)
        keys = pattern_keys(W)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        out = np.zeros(W.shape[0])
        for v, w in zip(self.vertices, self.weights, strict=True):
            key = pattern_keys(v[None, :])[0]
            pos = int(np.searchsorted(sorted_keys, key))
            if pos < len(keys) and sorted_keys[pos] == key:
                out[order[pos]] += w
        return out


def _as_oracle(source: np.ndarray | VertexOracle) -> VertexOracle:
    if callable(source):
        return source
    return EnumeratedVertices(np.asarray(source, dtype=float))


def _check_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q must be a non-empty 1-D vector")
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("forecasts must be finite and lie in [0, 1]")
    return q


def _affine_minimiser(A: np.ndarray) -> np.ndarray:
    """Barycentric weights of the minimum-norm point of ``aff(rows of A)``.

    Solves the bordered system ``[G 1; 1ᵀ 0] [μ; ν] = [0; 1]`` with ``G = A Aᵀ``; it is
    non-singular exactly when the rows of ``A`` are affinely independent.
    """
    s = A.shape[0]
    kkt = np.zeros((s + 1, s + 1))
    kkt[:s, :s] = A @ A.T
    kkt[:s, s] = 1.0
    kkt[s, :s] = 1.0
    rhs = np.zeros(s + 1)
    rhs[s] = 1.0
    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    return sol[:s]


def euclidean_projection(
    source: np.ndarray | VertexOracle,
    q: np.ndarray,
    tol: float = 1e-12,
    max_iter: int | None = None,
) -> Projection:
    """Euclidean projection of ``q`` onto ``conv`` of the vertices (Wolfe 1976).

    Computes the minimum-norm point of ``C - q`` by Wolfe's algorithm: a *major cycle*
    adds the vertex returned by the oracle, *minor cycles* move to the minimum-norm point
    of the affine hull of the current corral, dropping vertices whose weight hits zero.
    It stops when the Frank-Wolfe gap of ``f(p) = ‖p - q‖²``, namely
    ``2 (‖x‖² - min_v <x, v - q>)``, is at most ``tol``.

    Parameters
    ----------
    source
        Vertex matrix ``(k, m)`` or a :class:`VertexOracle`.
    q
        Forecast vector in ``[0, 1]^m``.
    tol
        Gap tolerance (absolute, in squared-Euclidean units); ``f(p) - f* ≤ gap``.
    """
    q = _check_q(q)
    oracle = _as_oracle(source)
    m = q.size
    limit = max_iter if max_iter is not None else 200 * (m + 10)

    first = np.asarray(oracle(-q), dtype=float)
    A = (first - q)[None, :]
    lam = np.ones(1)
    gap = np.inf
    it = 1
    while it <= limit:
        x = lam @ A
        v = np.asarray(oracle(x), dtype=float)
        p = v - q
        it += 1
        gap = 2.0 * float(x @ x - x @ p)
        if gap <= tol:
            break
        if (np.abs(A - p).max(axis=1) == 0.0).any():
            break
        A = np.vstack([A, p])
        lam = np.append(lam, 0.0)
        while True:
            mu = _affine_minimiser(A)
            if (mu > 1e-14).all():
                lam = mu / mu.sum()
                break
            below = mu <= 1e-14
            denom = lam[below] - mu[below]
            ratio = np.divide(lam[below], denom, out=np.zeros(denom.shape), where=denom > 0)
            theta = float(ratio.min())
            lam = lam + theta * (mu - lam)
            keep = lam > 1e-14
            if keep.all():
                keep[int(np.argmin(lam))] = False
            A, lam = A[keep], lam[keep]
            lam = lam / lam.sum()
            if A.shape[0] == 1:
                break
    x = lam @ A
    return Projection(np.clip(q + x, 0.0, 1.0), q, A + q, lam, max(gap, 0.0), it)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """``Σ_j KL(Bernoulli(p_j) ‖ Bernoulli(q_j))`` (the log-score Bregman divergence)."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    q = np.asarray(q, dtype=float)
    return float(np.sum(rel_entr(p, q) + rel_entr(1.0 - p, 1.0 - q)))


def _kl_grad(p: np.ndarray, q: np.ndarray, free: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-15, 1.0 - 1e-15)
    g = np.zeros_like(p)
    g[free] = np.log(pc[free] * (1.0 - q[free]) / ((1.0 - pc[free]) * q[free]))
    return g


def _line_search(deriv: Callable[[float], float], gamma_max: float) -> float:
    """Minimiser of a convex 1-D function on ``[0, gamma_max]`` from its derivative."""
    if deriv(0.0) >= 0.0:
        return 0.0
    if deriv(gamma_max) <= 0.0:
        return gamma_max
    return float(brentq(deriv, 0.0, gamma_max, xtol=1e-16, rtol=1e-14, maxiter=200))


def kl_projection(
    source: np.ndarray | VertexOracle,
    q: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-10,
    max_iter: int = 100_000,
) -> Projection:
    """KL projection ``argmin_{p ∈ C} Σ_j KL(p_j ‖ q_j)`` by away-step Frank-Wolfe.

    ``q`` is clipped to ``[eps, 1 - eps]`` so that the log score is finite in every world;
    the clipped vector is returned in :attr:`Projection.q`. The method starts at the
    barycentre of the extreme vertices along every coordinate (interior in each varying
    coordinate, where the KL gradient is finite) and uses exact line search on the
    monotone directional derivative. It stops when the Frank-Wolfe gap
    ``<∇f(p), p - s>`` (an upper bound on ``f(p) - f*``) is at most ``tol``.
    """
    q = np.clip(_check_q(q), eps, 1.0 - eps)
    oracle = _as_oracle(source)
    m = q.size
    eye = np.eye(m)
    lows = np.array([oracle(eye[j]) for j in range(m)])
    highs = np.array([oracle(-eye[j]) for j in range(m)])
    free = lows[np.arange(m), np.arange(m)] != highs[np.arange(m), np.arange(m)]
    seeds = np.unique(np.vstack([lows, highs]), axis=0)
    V = seeds
    lam = np.full(V.shape[0], 1.0 / V.shape[0])
    p = lam @ V
    gap = np.inf
    it = 0
    for _ in range(max_iter):
        it += 1
        g = _kl_grad(p, q, free)
        s = np.asarray(oracle(g), dtype=float)
        gap_fw = float(g @ (p - s))
        gap = gap_fw
        if gap_fw <= tol:
            break
        scores = V @ g
        ia = int(np.argmax(np.where(lam > 0, scores, -np.inf)))
        gap_away = float(scores[ia] - g @ p)
        if gap_fw >= gap_away:
            d, gamma_max, away = s - p, 1.0, False
        else:
            d, gamma_max, away = p - V[ia], lam[ia] / (1.0 - lam[ia]), True
        gamma = _line_search(lambda t, d=d, p=p: float(_kl_grad(p + t * d, q, free) @ d), gamma_max)
        if gamma <= 0.0:
            break
        if away:
            lam = lam * (1.0 + gamma)
            lam[ia] -= gamma
            if lam[ia] <= 1e-14:
                V, lam = np.delete(V, ia, axis=0), np.delete(lam, ia)
        else:
            lam = lam * (1.0 - gamma)
            hit = np.where((s == V).all(axis=1))[0]
            if hit.size:
                lam[hit[0]] += gamma
            else:
                V, lam = np.vstack([V, s]), np.append(lam, gamma)
        lam = lam / lam.sum()
        p = lam @ V
    return Projection(np.clip(p, 0.0, 1.0), q, V, lam, max(gap, 0.0), it)

"""Coherent extension bounds (de Finetti's fundamental theorem of prevision).

Given coherent forecasts ``q`` for formulas ``φ_1, ..., φ_m`` and a further formula ``ψ``,
the set of values ``P(ψ)`` that keep the enlarged system coherent is a closed interval

    [ min_λ tᵀλ , max_λ tᵀλ ],   λ ∈ Δ,  Wᵀλ = q,

with ``t`` the truth vector of ``ψ`` over the worlds. Both ends are linear programs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from dutchbook.coherence import DEFAULT_TOL, check_forecasts, coherence_distance
from dutchbook.logic import Formula, WorldSpace, as_formula, world_space

_HIGHS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}


class IncoherentForecastsError(ValueError):
    """The forecasts admit no coherent extension because they are already incoherent."""

    def __init__(self, distance: float) -> None:
        super().__init__(
            f"forecasts are incoherent (sup-norm distance {distance:.3g} to the coherent set); "
            "repair them first or audit for the Dutch book"
        )
        self.distance = distance


@dataclass(frozen=True)
class ExtensionBounds:
    """Interval of coherent probabilities for a new formula.

    ``lower_measure`` and ``upper_measure`` are probability measures over the distinct
    worlds of ``space`` that attain the two ends.
    """

    lower: float
    upper: float
    lower_measure: np.ndarray
    upper_measure: np.ndarray
    space: WorldSpace

    @property
    def width(self) -> float:
        return self.upper - self.lower


def extension_bounds(
    formulas: Sequence[Formula | str],
    forecasts: Sequence[float] | np.ndarray,
    target: Formula | str,
    tol: float = DEFAULT_TOL,
) -> ExtensionBounds:
    """Interval of probabilities for ``target`` consistent with coherent ``forecasts``.

    Raises
    ------
    IncoherentForecastsError
        If ``forecasts`` lie further than ``tol`` from the coherent polytope.
    """
    fs = [as_formula(f) for f in formulas]
    psi = as_formula(target)
    space = world_space([*fs, psi])
    m = len(fs)
    q = check_forecasts(forecasts, m)
    W = space.W[:, :m]
    t = space.W[:, m]
    dist = coherence_distance(W, q, "linf").value
    if dist > tol:
        raise IncoherentForecastsError(dist)
    k = W.shape[0]
    Wt = sparse.csr_matrix(W.T)
    ones = sparse.csr_matrix(np.ones((1, k)))
    out = []
    for sign in (1.0, -1.0):
        res = linprog(
            sign * t,
            A_eq=sparse.vstack([Wt, ones], format="csr"),
            b_eq=np.concatenate([q, [1.0]]),
            bounds=[(0, None)] * k,
            method="highs",
            options=_HIGHS,
        )
        if res.status == 2 and dist > 0.0:
            band = 2.0 * dist + 1e-12
            res = linprog(
                sign * t,
                A_ub=sparse.vstack([Wt, -Wt], format="csr"),
                b_ub=np.concatenate([q + band, -(q - band)]),
                A_eq=ones,
                b_eq=[1.0],
                bounds=[(0, None)] * k,
                method="highs",
                options=_HIGHS,
            )
        if not res.success:
            raise RuntimeError(f"extension LP failed: {res.message}")
        out.append((float(sign * res.fun), np.clip(res.x, 0.0, None)))
    (lo, lam_lo), (hi, lam_hi) = out[0], out[1]
    return ExtensionBounds(max(lo, 0.0), min(hi, 1.0), lam_lo, lam_hi, space)

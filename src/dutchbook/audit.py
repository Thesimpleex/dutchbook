"""High-level audit and repair of forecasts attached to formulas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dutchbook.coherence import (
    DEFAULT_TOL,
    DutchBook,
    StakeNorm,
    check_forecasts,
    coherence_distance,
    find_dutch_book,
)
from dutchbook.dominance import DominanceCertificate, dominance_certificate
from dutchbook.logic import Formula, WorldSpace, as_formula, collect_atoms, world_space
from dutchbook.oracle import MilpOracle, find_dutch_book_cg
from dutchbook.projection import Projection, euclidean_projection, kl_projection

Divergence = Literal["brier", "kl"]
Method = Literal["auto", "enumerate", "column-generation"]
ENUMERATION_ATOMS = 16
"""``method="auto"`` enumerates worlds up to this many atoms and generates columns beyond."""


@dataclass(frozen=True)
class Audit:
    """Coherence audit of one forecast set.

    Attributes
    ----------
    space : WorldSpace
        Distinct worlds and truth matrix of the formulas.
    labels : tuple of str
        Display name of every formula.
    forecasts : ndarray
        The audited forecasts.
    book : DutchBook
        Optimal Dutch book for the requested stake norm (all-zero stakes if coherent).
    distances : dict
        Distance of the forecasts to the coherent polytope in the ℓ∞, ℓ1 and ℓ2 norms
        (only the norms that were computed when worlds are generated, not enumerated).
    exhaustive : bool
        True when ``space`` lists every distinct world. With column generation ``space``
        holds only the generated worlds; the guaranteed profit is still certified for all.
    """

    space: WorldSpace
    labels: tuple[str, ...]
    forecasts: np.ndarray
    book: DutchBook
    distances: dict[str, float]
    exhaustive: bool = True

    @property
    def coherent(self) -> bool:
        return self.book.coherent


@dataclass(frozen=True)
class Repair:
    """Coherentized forecasts with their accuracy-dominance certificate."""

    space: WorldSpace
    labels: tuple[str, ...]
    forecasts: np.ndarray
    repaired: np.ndarray
    divergence: Divergence
    projection: Projection
    certificate: DominanceCertificate


def _prepare(
    formulas: Sequence[Formula | str],
    forecasts: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None,
    atoms: Sequence[str] | None,
) -> tuple[WorldSpace, np.ndarray, tuple[str, ...]]:
    fs = [as_formula(f) for f in formulas]
    space = world_space(fs, atoms)
    q = check_forecasts(forecasts, len(fs))
    names = tuple(labels) if labels is not None else tuple(str(f) for f in fs)
    if len(names) != len(fs):
        raise ValueError("labels must match formulas in length")
    return space, q, names


def audit(
    formulas: Sequence[Formula | str],
    forecasts: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    stake_norm: StakeNorm = "l1",
    atoms: Sequence[str] | None = None,
    tol: float = DEFAULT_TOL,
    method: Method = "auto",
) -> Audit:
    """Test ``forecasts`` for coherence and price the Dutch book against them.

    Parameters
    ----------
    formulas
        One formula (or string) per forecast.
    forecasts
        Probabilities in ``[0, 1]``.
    stake_norm
        ``"l1"``: total stake ≤ 1 (profit = sup-norm distance to coherence, as in
        Andrews and Sarkar 2026); ``"linf"``: every stake ≤ 1 (ℓ1 distance);
        ``"l2"``: Euclidean stake norm (ℓ2 distance).
    method
        ``"enumerate"`` lists all worlds (up to 22 atoms); ``"column-generation"`` prices
        the Dutch book with a MILP oracle and scales to many atoms; ``"auto"`` enumerates
        up to 16 atoms.
    """
    if _use_column_generation(formulas, atoms, method):
        return _audit_cg(formulas, forecasts, labels, stake_norm, atoms, tol)
    space, q, names = _prepare(formulas, forecasts, labels, atoms)
    book = find_dutch_book(space.W, q, stake_norm, tol)
    distances = {n: coherence_distance(space.W, q, n).value for n in ("linf", "l1", "l2")}
    return Audit(space, names, q, book, distances)


def repair(
    formulas: Sequence[Formula | str],
    forecasts: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    divergence: Divergence = "brier",
    atoms: Sequence[str] | None = None,
    tol: float = DEFAULT_TOL,
    eps: float = 1e-6,
    method: Method = "auto",
) -> Repair:
    """Project forecasts onto the coherent polytope and certify the improvement.

    ``divergence="brier"`` is the Euclidean projection and guarantees
    ``Brier(q, w) - Brier(p, w) ≥ ‖p - q‖²`` in every world ``w``; ``"kl"`` is the KL
    projection with the analogous log-score guarantee (``q`` clipped to
    ``[eps, 1 - eps]``). The guarantee concerns the whole forecast vector.
    """
    if divergence not in ("brier", "kl"):
        raise ValueError("divergence must be 'brier' or 'kl'")
    generated = _use_column_generation(formulas, atoms, method)
    if generated:
        fs = [as_formula(f) for f in formulas]
        q = check_forecasts(forecasts, len(fs))
        names = tuple(labels) if labels is not None else tuple(str(f) for f in fs)
        oracle = MilpOracle(fs, atoms)
        proj = (
            euclidean_projection(oracle, q)
            if divergence == "brier"
            else kl_projection(oracle, q, eps=eps)
        )
        assignments = np.array([oracle.solve(_pin(v))[1] for v in proj.vertices])
        space = WorldSpace(tuple(fs), oracle.atoms, proj.vertices, assignments)
    else:
        space, q, names = _prepare(formulas, forecasts, labels, atoms)
        proj = (
            euclidean_projection(space.W, q)
            if divergence == "brier"
            else kl_projection(space.W, q, eps=eps)
        )
    scoring: Literal["brier", "log"] = "brier" if divergence == "brier" else "log"
    cert = dominance_certificate(space.W, proj.q, proj.point, scoring, proj.gap, tol)
    return Repair(space, names, q, proj.point, divergence, proj, cert)


def _pin(vertex: np.ndarray) -> np.ndarray:
    """Direction whose unique minimiser is ``vertex``: reward agreement with it."""
    return np.where(vertex > 0.5, -1.0, 1.0)


def _use_column_generation(
    formulas: Sequence[Formula | str], atoms: Sequence[str] | None, method: Method
) -> bool:
    if method not in ("auto", "enumerate", "column-generation"):
        raise ValueError("method must be 'auto', 'enumerate' or 'column-generation'")
    if method != "auto":
        return method == "column-generation"
    n = len(atoms) if atoms is not None else len(collect_atoms(as_formula(f) for f in formulas))
    return n > ENUMERATION_ATOMS


def _audit_cg(
    formulas: Sequence[Formula | str],
    forecasts: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None,
    stake_norm: StakeNorm,
    atoms: Sequence[str] | None,
    tol: float,
) -> Audit:
    fs = [as_formula(f) for f in formulas]
    q = check_forecasts(forecasts, len(fs))
    names = tuple(labels) if labels is not None else tuple(str(f) for f in fs)
    if len(names) != len(fs):
        raise ValueError("labels must match formulas in length")
    oracle = MilpOracle(fs, atoms)
    if stake_norm == "l2":
        proj = euclidean_projection(oracle, q)
        dist = float(np.linalg.norm(proj.point - q))
        assignments = np.array([oracle.solve(_pin(v))[1] for v in proj.vertices])
        if dist <= tol:
            book = DutchBook(
                np.zeros(len(fs)), np.zeros(len(proj.vertices)), 0.0, "l2", proj.weights, True
            )
        else:
            stakes = (proj.point - q) / dist
            payoffs = (proj.vertices - q) @ stakes
            # margin >= -gap/2 in every world (Frank-Wolfe gap of the squared distance)
            profit = max(dist - proj.gap / (2.0 * dist), 0.0)
            book = DutchBook(stakes, payoffs, profit, "l2", proj.weights, False)
        space = WorldSpace(tuple(fs), oracle.atoms, proj.vertices, assignments)
        return Audit(space, names, q, book, {"l2": dist}, exhaustive=False)
    cg = find_dutch_book_cg(fs, q, stake_norm, tol, oracle=oracle)
    proj = euclidean_projection(oracle, q)
    space = WorldSpace(tuple(fs), oracle.atoms, cg.worlds, cg.assignments)
    key = "linf" if stake_norm == "l1" else "l1"
    distances = {key: cg.book.profit, "l2": float(np.linalg.norm(proj.point - q))}
    return Audit(space, names, q, cg.book, distances, exhaustive=False)

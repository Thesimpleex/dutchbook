"""Coherence audit of ForecastBench combination tuples.

For every tuple ``q = (P(A), P(B), P(A∧B), P(A∧¬B), P(¬A∧B), P(¬A∧¬B))`` this computes

* the guaranteed Dutch-book profit per unit of total stake (ℓ∞ distance to the coherent
  polytope; exact LP), the Euclidean distance, and the Brier projection;
* the same two quantities after the *renormalisation baseline* that rescales the four
  joint cells to sum to one and keeps the marginals, which removes the part of the
  incoherence that is mechanically caused by eliciting each cell in a separate prompt;
* the per-world Pythagorean margin ``Brier(q, w) - Brier(p, w) - ‖p - q‖²`` in all four
  worlds, which must be non-negative.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from dutchbook.coherence import coherence_distance
from dutchbook.dominance import dominance_certificate
from dutchbook.forecastbench import FORMULAS, TupleTable
from dutchbook.logic import world_space
from dutchbook.projection import euclidean_projection

MATERIAL_THRESHOLDS = (0.01, 0.05, 0.10)
"""Guaranteed profit per unit of total stake treated as material (beyond rounding noise)."""
TOL = 1e-9


@lru_cache(maxsize=1)
def tuple_worlds() -> np.ndarray:
    """Truth matrix (4 worlds x 6 formulas) of a combination tuple."""
    return world_space(FORMULAS).W


def renormalise_cells(q: np.ndarray) -> np.ndarray:
    """Rescale the four joint cells to sum to one; the marginals are unchanged.

    Tuples whose cells sum to zero get the uniform joint ``(1/4, 1/4, 1/4, 1/4)``.
    """
    out = np.array(q, dtype=float, copy=True)
    cells = out[:, 2:]
    total = cells.sum(axis=1, keepdims=True)
    uniform = np.full_like(cells, 0.25)
    out[:, 2:] = np.where(total > 0, cells / np.where(total > 0, total, 1.0), uniform)
    return out


def _solve_rows(rows: np.ndarray) -> np.ndarray:
    """Per row: ℓ∞ distance, ℓ2 distance, projection (6), min Pythagorean margin, gap."""
    W = tuple_worlds()
    out = np.empty((rows.shape[0], 10))
    for i, q in enumerate(rows):
        linf = coherence_distance(W, q, "linf").value
        proj = euclidean_projection(W, q)
        cert = dominance_certificate(W, q, proj.point, "brier", proj.gap, TOL)
        out[i, 0] = linf
        out[i, 1] = float(np.linalg.norm(proj.point - q))
        out[i, 2:8] = proj.point
        out[i, 8] = float(cert.margin.min())
        out[i, 9] = proj.gap
    return out


def solve_unique(
    q: np.ndarray,
    workers: int | None = None,
    chunk: int = 4000,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Solve :func:`_solve_rows` once per distinct row of ``q`` and map back to all rows."""
    uniq, inverse = np.unique(q, axis=0, return_inverse=True)
    chunks = [uniq[i : i + chunk] for i in range(0, len(uniq), chunk)]
    if workers == 1 or len(chunks) == 1:
        parts = []
        for k, c in enumerate(chunks):
            parts.append(_solve_rows(c))
            if progress:
                progress(k + 1, len(chunks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = []
            for k, part in enumerate(pool.map(_solve_rows, chunks)):
                parts.append(part)
                if progress:
                    progress(k + 1, len(chunks))
    solved = np.vstack(parts)
    return solved[inverse.reshape(-1)]


@dataclass(frozen=True)
class TupleMetrics:
    """Per-tuple results (aligned with the rows of the :class:`TupleTable`)."""

    linf: np.ndarray
    l2: np.ndarray
    projected: np.ndarray
    margin_min: np.ndarray
    gap: np.ndarray
    renormalised: np.ndarray
    linf_renorm: np.ndarray
    l2_renorm: np.ndarray
    margin_min_renorm: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> TupleMetrics:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    def select(self, mask: np.ndarray) -> TupleMetrics:
        return TupleMetrics(**{k: v[mask] for k, v in self.__dict__.items()})


def compute_metrics(
    table: TupleTable,
    workers: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TupleMetrics:
    """Audit every tuple of ``table`` (raw forecasts and the renormalised baseline)."""
    raw = solve_unique(
        table.q, workers, progress=(lambda k, n: progress("raw", k, n)) if progress else None
    )
    renorm_q = renormalise_cells(table.q)
    ren = solve_unique(
        renorm_q,
        workers,
        progress=(lambda k, n: progress("renormalised", k, n)) if progress else None,
    )
    return TupleMetrics(
        linf=raw[:, 0],
        l2=raw[:, 1],
        projected=raw[:, 2:8],
        margin_min=raw[:, 8],
        gap=raw[:, 9],
        renormalised=renorm_q,
        linf_renorm=ren[:, 0],
        l2_renorm=ren[:, 1],
        margin_min_renorm=ren[:, 8],
    )


def cached_metrics(
    table: TupleTable,
    rounds: list[str],
    cache_dir: str | Path,
    digest: str,
    workers: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> TupleMetrics:
    """Metrics for the tuples of ``rounds``, cached per round under ``cache_dir``.

    The result has one row per row of ``table``; rows of other rounds are ``nan``.
    """
    fields = TupleMetrics.__dataclass_fields__
    n = len(table)
    full = {k: np.full((n, 6) if k in ("projected", "renormalised") else n, np.nan) for k in fields}
    for r in rounds:
        mask = table.round_idx == table.rounds.index(r)
        path = Path(cache_dir) / f"metrics-{digest[:16]}-{r}.npz"
        if path.exists():
            part = TupleMetrics.load(path)
        else:
            part = compute_metrics(table.select(mask), workers, progress)
            path.parent.mkdir(parents=True, exist_ok=True)
            part.save(path)
        for k in fields:
            full[k][mask] = getattr(part, k)
    return TupleMetrics(**full)


def brier(q: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Mean squared error over the six coordinates, per tuple."""
    return ((q - outcome) ** 2).mean(axis=1)


def brier_cells(q: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Mean squared error over the four joint cells, per tuple."""
    return ((q[:, 2:] - outcome[:, 2:]) ** 2).mean(axis=1)


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    reps: int = 2000,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Mean with a 95% percentile interval from a bootstrap over clusters (question pairs).

    ``values`` may be a boolean indicator (share). With ``weights`` the statistic is the
    weighted mean ``Σ w v / Σ w`` (used for ratio-type statistics).
    """
    values = np.asarray(values, dtype=float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    _, inv = np.unique(clusters, return_inverse=True)
    n_c = int(inv.max()) + 1
    num = np.bincount(inv, weights=values * w, minlength=n_c)
    den = np.bincount(inv, weights=w, minlength=n_c)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(reps, n_c))
    stats = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def _pct(t: float) -> str:
    return f"{round(t * 100)}pct"


def is_independence_constructed(q: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Tuples whose four joint cells equal the products of the two marginals.

    Such tuples are coherent by construction (the joint was computed from the marginals
    under independence rather than elicited), so they say nothing about a model's beliefs.
    """
    a, b = q[:, 0], q[:, 1]
    product = np.stack([a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)], axis=1)
    return (np.abs(q[:, 2:] - product) <= atol).all(axis=1)


def split_model_name(name: str) -> tuple[str, str]:
    """``"Model-X (zero shot)"`` to ``("Model-X", "zero shot")``."""
    if name.endswith(")") and " (" in name:
        base, _, variant = name[:-1].partition(" (")
        return base, variant
    return name, ""


def summarise(table: TupleTable, metrics: TupleMetrics, mask: np.ndarray) -> dict[str, float]:
    """Incoherence and repair statistics for the tuples selected by ``mask``."""
    q = table.q[mask]
    linf, linf_r = metrics.linf[mask], metrics.linf_renorm[mask]
    out: dict[str, float] = {
        "n_tuples": int(mask.sum()),
        "n_rounds": int(np.unique(table.round_idx[mask]).size),
        "mean_sum_joint_cells": float(q[:, 2:].sum(axis=1).mean()),
        "share_cells_sum_to_one": float((np.abs(q[:, 2:].sum(axis=1) - 1) <= 1e-9).mean()),
        "share_independence_constructed": float(is_independence_constructed(q).mean()),
        "share_dutch_bookable": float((linf > TOL).mean()),
        "mean_arbitrage": float(linf.mean()),
        "median_arbitrage": float(np.median(linf)),
        "p90_arbitrage": float(np.percentile(linf, 90)),
        "share_dutch_bookable_after_renorm": float((linf_r > TOL).mean()),
        "mean_arbitrage_after_renorm": float(linf_r.mean()),
        "median_arbitrage_after_renorm": float(np.median(linf_r)),
        "mean_l2_distance": float(metrics.l2[mask].mean()),
        "mean_l2_distance_after_renorm": float(metrics.l2_renorm[mask].mean()),
    }
    for t in MATERIAL_THRESHOLDS:
        out[f"share_arbitrage_ge_{_pct(t)}"] = float((linf >= t).mean())
        out[f"share_arbitrage_ge_{_pct(t)}_after_renorm"] = float((linf_r >= t).mean())
    slack = metrics.margin_min[mask] + metrics.gap[mask] + TOL
    out["dominance_violations"] = int((slack < 0).sum())
    out["min_dominance_margin"] = float(metrics.margin_min[mask].min())
    resolved = mask & table.resolved
    n_res = int(resolved.sum())
    out["n_resolved"] = n_res
    if n_res:
        o = table.outcome[resolved]
        qr, pr, nr = table.q[resolved], metrics.projected[resolved], metrics.renormalised[resolved]
        b_raw, b_proj, b_norm = brier(qr, o), brier(pr, o), brier(nr, o)
        c_raw, c_proj, c_norm = brier_cells(qr, o), brier_cells(pr, o), brier_cells(nr, o)
        out.update(
            {
                "brier_raw": float(b_raw.mean()),
                "brier_projected": float(b_proj.mean()),
                "brier_renormalised": float(b_norm.mean()),
                "brier_cells_raw": float(c_raw.mean()),
                "brier_cells_projected": float(c_proj.mean()),
                "brier_cells_renormalised": float(c_norm.mean()),
                "share_projection_improves": float((b_proj < b_raw - 1e-12).mean()),
                "share_projection_worsens": float((b_proj > b_raw + 1e-12).mean()),
                "share_cells_worsen_under_projection": float((c_proj > c_raw + 1e-12).mean()),
                "share_renormalisation_worsens": float((b_norm > b_raw + 1e-12).mean()),
            }
        )
    return out


def model_rows(table: TupleTable, metrics: TupleMetrics) -> list[dict[str, object]]:
    """One summary row per model configuration (``kind`` ``model`` or ``reference``)."""
    rows = []
    for i, (org, name, kind) in enumerate(table.models):
        mask = table.model_idx == i
        if not mask.any():
            continue
        base, variant = split_model_name(name)
        row: dict[str, object] = {
            "organization": org,
            "model": name,
            "base_model": base,
            "variant": variant,
            "kind": kind,
        }
        row.update(summarise(table, metrics, mask))
        rows.append(row)
    return rows


INDEPENDENCE_SHARE = 0.9
"""Configurations whose joint cells equal the independent product in more than this share
of tuples are excluded from model summaries (their coherence is by construction)."""


def overall_summary(table: TupleTable, metrics: TupleMetrics) -> dict[str, float]:
    """Pooled statistics over all model configurations with elicited joint cells."""
    include = np.zeros(len(table), dtype=bool)
    n_models = 0
    for i, (_, _, kind) in enumerate(table.models):
        mask = table.model_idx == i
        if kind != "model" or not mask.any():
            continue
        if is_independence_constructed(table.q[mask]).mean() > INDEPENDENCE_SHARE:
            continue
        include |= mask
        n_models += 1
    out = summarise(table, metrics, include)
    out["n_models"] = n_models
    return out

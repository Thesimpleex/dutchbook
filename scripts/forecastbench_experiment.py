"""ForecastBench combination questions: download, audit every tuple, write the results.

Pipeline: download the processed forecast sets into ``data/raw`` (gitignored), extract the
complete tuples, solve the exact Dutch-book LP and the Brier projection for every tuple
(and for the renormalised baseline), aggregate per model configuration, per round and
overall (with cluster-bootstrap intervals over question pairs), choose a representative
tuple for the README, and write the ``forecastbench`` section of ``docs/results.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json

import numpy as np
from _common import CACHE, RAW, ROOT, log, update_results

from dutchbook import audit, find_dutch_book
from dutchbook import benchmark as bm
from dutchbook import forecastbench as fb

HERO_ROUND = "2024-07-21"
HERO_MODEL = "GPT-4o (zero shot)"
MARKET_SOURCES = ("manifold", "metaculus", "polymarket", "infer")
SEED = 20261004


def ecdf_points(x: np.ndarray, grid: np.ndarray) -> list[float]:
    xs = np.sort(x)
    return [float(np.searchsorted(xs, g, side="right") / xs.size) for g in grid]


def included_mask(table: fb.TupleTable) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Tuples of model configurations whose joint cells were elicited (not derived)."""
    mask = np.zeros(len(table), dtype=bool)
    excluded = []
    for i, (org, name, kind) in enumerate(table.models):
        m = table.model_idx == i
        if kind != "model" or not m.any():
            continue
        share = float(bm.is_independence_constructed(table.q[m]).mean())
        if share > bm.INDEPENDENCE_SHARE:
            excluded.append(
                {
                    "organization": org,
                    "model": name,
                    "n_tuples": int(m.sum()),
                    "share_independence_constructed": share,
                }
            )
            continue
        mask |= m
    return mask, excluded


def headline_intervals(table: fb.TupleTable, metrics: bm.TupleMetrics, mask: np.ndarray) -> dict:
    """Pooled means with 95% cluster-bootstrap intervals (clusters: question pairs)."""
    cl = table.pair[mask]
    res = mask & table.resolved
    out = {}
    stats = {
        "mean_arbitrage": (metrics.linf[mask], cl),
        "mean_arbitrage_after_renorm": (metrics.linf_renorm[mask], cl),
        "share_arbitrage_ge_5pct": ((metrics.linf[mask] >= 0.05).astype(float), cl),
        "share_arbitrage_ge_5pct_after_renorm": (
            (metrics.linf_renorm[mask] >= 0.05).astype(float),
            cl,
        ),
    }
    o = table.outcome[res]
    b_raw = bm.brier(table.q[res], o)
    b_proj = bm.brier(metrics.projected[res], o)
    b_norm = bm.brier(metrics.renormalised[res], o)
    cr = table.pair[res]
    stats.update(
        {
            "brier_raw": (b_raw, cr),
            "brier_projected": (b_proj, cr),
            "brier_renormalised": (b_norm, cr),
            "brier_gain_projection": (b_raw - b_proj, cr),
            "brier_gain_renormalisation": (b_raw - b_norm, cr),
        }
    )
    for k, (v, c) in stats.items():
        mean, lo, hi = bm.cluster_bootstrap(v, c, reps=1000, seed=SEED)
        out[k] = {"mean": mean, "ci95": [lo, hi]}
    return out


def repair_path(table: fb.TupleTable, metrics: bm.TupleMetrics, mask: np.ndarray) -> dict:
    """Pooled Brier score along the segment from each forecast to its projection."""
    res = mask & table.resolved
    q, p, o = table.q[res], metrics.projected[res], table.outcome[res]
    base = bm.brier(q, o)
    ts = np.linspace(0.0, 1.0, 21)
    whole, cells, worse = [], [], []
    for t in ts:
        qt = q + t * (p - q)
        b = bm.brier(qt, o)
        whole.append(float(b.mean()))
        cells.append(float(bm.brier_cells(qt, o).mean()))
        worse.append(int((b > base + 1e-12).sum()))
    return {
        "t": [float(t) for t in ts],
        "brier": whole,
        "brier_cells": cells,
        "tuples_worse_than_unrepaired": worse,
        "n_tuples": int(res.sum()),
    }


def lp_cross_check(
    table: fb.TupleTable, metrics: bm.TupleMetrics, mask: np.ndarray, n: int = 20000
) -> dict:
    """Independent checks of the LP results on real tuples.

    The distance LP (used for every tuple) is compared with the bettor's primal LP on a
    random sample (strong duality), and on all tuples the sup-norm distance must sit between
    ``l2 / sqrt(6)`` and ``l2`` (norm equivalence, valid for the distances to a convex set).
    """
    rng = np.random.default_rng(SEED)
    pool = np.where(mask)[0]
    sample = rng.choice(pool, size=min(n, pool.size), replace=False)
    W = bm.tuple_worlds()
    worst = 0.0
    for i in sample:
        book = find_dutch_book(W, table.q[i], "l1")
        worst = max(worst, abs(book.profit - metrics.linf[i]))
    linf, l2 = metrics.linf[mask], metrics.l2[mask]
    return {
        "sample_size": int(sample.size),
        "max_abs_difference_bettor_lp_vs_distance_lp": float(worst),
        "tuples_outside_l2_sandwich": int(
            ((linf > l2 + 1e-9) | (linf < l2 / np.sqrt(6.0) - 1e-9)).sum()
        ),
    }


def pick_hero(table: fb.TupleTable, metrics: bm.TupleMetrics) -> int:
    """Median-arbitrage resolved market-question tuple of the hero model in the hero round."""
    mi = next(i for i, m in enumerate(table.models) if m[1] == HERO_MODEL)
    ri = table.rounds.index(HERO_ROUND)
    ok = (
        (table.model_idx == mi)
        & (table.round_idx == ri)
        & table.resolved
        & np.isin(table.source, MARKET_SOURCES)
    )
    idx = np.where(ok)[0]
    order = idx[np.argsort(metrics.linf[idx], kind="stable")]
    return int(order[len(order) // 2])


def write_hero(table: fb.TupleTable, metrics: bm.TupleMetrics, row: int, raw) -> dict:
    texts = fb.question_texts(HERO_ROUND, str(table.pair[row]), raw)
    q = table.q[row]
    ids = ["A", "B", "A_and_B", "A_and_notB", "notA_and_B", "notA_and_notB"]
    exdir = ROOT / "examples"
    exdir.mkdir(exist_ok=True)
    with (exdir / "forecastbench_tuple.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "forecast"])
        for ident, v in zip(ids, q, strict=True):
            w.writerow([ident, f"{v:g}"])
    toml = [
        "# One combination tuple from ForecastBench (CC BY-SA 4.0, forecastbench.org).",
        f"# Forecaster: {HERO_MODEL}, round {HERO_ROUND}; question texts from the question set.",
        'title = "ForecastBench combination tuple"',
        "",
        "[atoms]",
        f"A = {json.dumps(texts[0], ensure_ascii=False)}",
        f"B = {json.dumps(texts[1], ensure_ascii=False)}",
        "",
        "[formulas]",
        *(f'{i} = "{f}"' for i, f in zip(ids, fb.FORMULAS, strict=True)),
        "",
    ]
    (exdir / "forecastbench_tuple.toml").write_text("\n".join(toml))
    a = audit(list(fb.FORMULAS), q, stake_norm="l1")
    qs_path = fb.download_question_set(HERO_ROUND, raw)
    return {
        "question_set": {
            "url": fb.QUESTION_SET_URL.format(round=HERO_ROUND),
            "sha256": hashlib.sha256(qs_path.read_bytes()).hexdigest(),
            "license": "CC BY-SA 4.0",
        },
        "round": HERO_ROUND,
        "model": HERO_MODEL,
        "selection": (
            "median guaranteed arbitrage among the model's resolved market-question tuples "
            "in the round"
        ),
        "pair": str(table.pair[row]),
        "resolution_date": str(table.resolution_date[row]),
        "questions": texts,
        "forecasts": dict(zip(ids, (float(v) for v in q), strict=True)),
        "guaranteed_profit": a.book.profit,
        "arbitrage_from_tuple_metrics": float(metrics.linf[row]),
        "sum_joint_cells": float(q[2:].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    log("downloading / checking snapshot")
    manifest = fb.download_snapshot(
        RAW,
        progress=lambda d, t: (
            log(f"  {d / 1e6:.0f} / {t / 1e6:.0f} MB") if d % (20 << 20) < (1 << 20) else None
        ),
    )
    log("reading tuples")
    table = fb.load_tuples(RAW, CACHE, progress=None)
    digest = fb.snapshot_digest(RAW)
    log(f"{len(table):,} tuples, {len(table.models)} configurations, {len(table.rounds)} rounds")
    metrics = bm.cached_metrics(
        table,
        list(table.rounds),
        CACHE,
        digest,
        args.workers,
        progress=lambda stage, k, n: log(f"  {stage} chunk {k}/{n}") if k % 20 == 0 else None,
    )
    rows = bm.model_rows(table, metrics)
    mask, excluded = included_mask(table)
    overall = bm.summarise(table, metrics, mask)
    overall["n_models"] = int(sum(1 for r in rows if r["kind"] == "model") - len(excluded))
    overall["ci95"] = headline_intervals(table, metrics, mask)
    rounds = []
    for k, r in enumerate(table.rounds):
        m = mask & (table.round_idx == k)
        s = bm.summarise(table, metrics, m)
        s["round"] = r
        rounds.append(s)
    by_source = {}
    for src in sorted(set(table.source[mask])):
        by_source[str(src)] = bm.summarise(table, metrics, mask & (table.source == src))
    variants = {}
    for v in sorted({str(r["variant"]) for r in rows if r["kind"] == "model"}):
        vm = np.zeros(len(table), dtype=bool)
        for i, (_, name, kind) in enumerate(table.models):
            if kind == "model" and bm.split_model_name(name)[1] == v:
                vm |= table.model_idx == i
        vm &= mask
        if vm.any():
            variants[v] = bm.summarise(table, metrics, vm)
    grid = np.linspace(0, 1, 201)
    ecdf = {
        "grid": [float(g) for g in grid],
        "raw": ecdf_points(metrics.linf[mask], grid),
        "renormalised": ecdf_points(metrics.linf_renorm[mask], grid),
    }
    hero_row = pick_hero(table, metrics)
    hero = write_hero(table, metrics, hero_row, RAW)
    resolved_dates = table.resolution_date[table.resolved]
    update_results(
        "forecastbench",
        {
            "snapshot": manifest,
            "data_vintage": {
                "downloaded": manifest.get("downloaded"),
                "last_forecast_due_date_with_combinations": table.rounds[-1],
                "last_resolution_date_observed": max(resolved_dates.tolist()),
            },
            "extraction": table.stats,
            "rounds": list(table.rounds),
            "overall": overall,
            "by_round": rounds,
            "by_source": by_source,
            "by_variant": variants,
            "excluded_configurations": excluded,
            "references": [r for r in rows if r["kind"] == "reference"],
            "models": sorted(
                (r for r in rows if r["kind"] == "model"), key=lambda r: r["mean_arbitrage"]
            ),
            "ecdf": ecdf,
            "hero_tuple": hero,
            "repair_path": repair_path(table, metrics, mask),
            "lp_cross_check": lp_cross_check(table, metrics, mask),
            "rounding_note": {
                "share_arbitrage_below_0.005": float((metrics.linf[mask] < 0.005).mean()),
                "share_arbitrage_below_0.005_after_renorm": float(
                    (metrics.linf_renorm[mask] < 0.005).mean()
                ),
            },
        },
    )
    log("done")


if __name__ == "__main__":
    main()

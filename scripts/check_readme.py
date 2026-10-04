"""Check that the numbers quoted in README.md match docs/results.json.

Every expected string is formatted from the results file; the script fails if one of them
is missing from the README, which happens when the results change and the text is stale.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import ROOT, load_results


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def main() -> int:
    res = load_results()
    fb = res["forecastbench"]
    o = fb["overall"]
    ci = o["ci95"]
    zero = sorted(
        (
            m
            for m in fb["models"]
            if m["variant"] == "zero shot"
            and m["n_tuples"] >= 1000
            and m["share_independence_constructed"] < 0.9
        ),
        key=lambda m: m["mean_arbitrage"],
    )
    mean_arb = np.array([m["mean_arbitrage"] for m in zero])
    renorm = np.array([m["mean_arbitrage_after_renorm"] for m in zero])
    resolved = o["n_resolved"]
    assert o["dominance_violations"] == 0
    assert o["share_projection_worsens"] == 0.0
    expected = {
        "tuples": f"{o['n_tuples']:,} tuples",
        "configurations": f"{o['n_models']} model configurations",
        "resolved": f"{resolved:,}",
        "rounds": f"{o['n_rounds']} rounds",
        "sum of cells": f"{o['mean_sum_joint_cells']:.2f}",
        "mean arbitrage": f"**{o['mean_arbitrage']:.3f}**",
        "mean arbitrage ci": f"{ci['mean_arbitrage']['ci95'][0]:.3f} to "
        f"{ci['mean_arbitrage']['ci95'][1]:.3f}",
        "renormalised mean": f"**{o['mean_arbitrage_after_renorm']:.3f}**",
        "renormalised ci": f"{ci['mean_arbitrage_after_renorm']['ci95'][0]:.3f} to "
        f"{ci['mean_arbitrage_after_renorm']['ci95'][1]:.3f}",
        "bookable": pct(o["share_dutch_bookable"], 2),
        "5pct": pct(o["share_arbitrage_ge_5pct"]),
        "10pct": pct(o["share_arbitrage_ge_10pct"]),
        "5pct renormalised": pct(o["share_arbitrage_ge_5pct_after_renorm"]),
        "5pct renormalised ci": f"{pct(ci['share_arbitrage_ge_5pct_after_renorm']['ci95'][0])} to "
        f"{pct(ci['share_arbitrage_ge_5pct_after_renorm']['ci95'][1])}",
        "10pct renormalised": pct(o["share_arbitrage_ge_10pct_after_renorm"]),
        "brier raw": f"**{o['brier_raw']:.4f}**",
        "brier projected": f"**{o['brier_projected']:.4f}**",
        "brier renormalised": f"{o['brier_renormalised']:.4f}",
        "brier gain": f"gain {ci['brier_gain_projection']['mean']:.4f}",
        "brier gain ci": f"{ci['brier_gain_projection']['ci95'][0]:.4f} to "
        f"{ci['brier_gain_projection']['ci95'][1]:.4f}",
        "cells raw": f"{o['brier_cells_raw']:.4f}",
        "cells projected": f"{o['brier_cells_projected']:.4f}",
        "cells worsen": pct(o["share_cells_worsen_under_projection"]),
        "renormalisation worsens": pct(o["share_renormalisation_worsens"]),
        "projection worsens": "in none",
        "zero shot count": f"{len(zero)} zero-shot configurations",
        "zero shot min": f"{mean_arb.min():.3f} ({zero[0]['model'].replace(' (zero shot)', '')}",
        "zero shot max": f"{mean_arb.max():.3f} ({zero[-1]['model'].replace(' (zero shot)', '')}",
        "zero shot median": f"median of {np.median(mean_arb):.3f}",
        "renormalised range": f"between {renorm.min():.3f} and {renorm.max():.3f}",
        "rounding raw": pct(fb["rounding_note"]["share_arbitrage_below_0.005"]),
        "rounding renormalised": pct(
            fb["rounding_note"]["share_arbitrage_below_0.005_after_renorm"]
        ),
        "download date": fb["snapshot"]["downloaded"],
        "last resolution": fb["data_vintage"]["last_resolution_date_observed"],
        "hero sum": f"{fb['hero_tuple']['sum_joint_cells']:.2f}",
        "violations": f"{o['n_tuples']:,} tuples and each of its four",
    }
    pw = res["pairwise_graphs"]["configurations"].values()
    forests = sum(c["forests"]["components"] for c in pw)
    cyclic = sum(c["with_cycles"]["components"] for c in pw)
    bad = sum(c["with_cycles"]["globally_incoherent"] for c in pw)
    skipped = sum(c["components_skipped_over_max_atoms"] for c in pw)
    assert sum(c["forests"]["globally_incoherent"] for c in pw) == 0
    expected.update(
        {
            "pairwise components": f"{forests + cyclic:,} components with at most "
            f"{res['pairwise_graphs']['max_atoms_per_component']} questions",
            "pairwise skipped": f"{skipped} larger components",
            "pairwise forests": f"**{forests:,} components that are trees admit no Dutch book "
            "at all**",
            "pairwise cycles": f"**{bad} of the {cyclic} components with a cycle "
            f"({pct(bad / cyclic)})**",
            "pairwise max profit": "up to "
            f"{max(c['with_cycles']['max_global_profit'] for c in pw):.3f}",
        }
    )
    readme = (ROOT / "README.md").read_text()
    missing = {k: v for k, v in expected.items() if v not in readme}
    for k, v in missing.items():
        print(f"missing in README: {k}: {v!r}")
    print(f"{len(expected) - len(missing)} of {len(expected)} quoted numbers match results.json")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

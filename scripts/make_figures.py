"""Render the README figures (light and dark) from ``docs/results.json``."""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from _common import FIGURES, RESULTS, log
from matplotlib.patches import Circle, Polygon

from dutchbook.viz import save_figure


def short(name: str) -> str:
    return name.replace(" (zero shot)", "")


def fig_incoherence_by_model(res: dict[str, Any]):
    fb = res["forecastbench"]
    rows = [r for r in fb["models"] if r["variant"] == "zero shot" and r["n_tuples"] >= 1000]
    rows.sort(key=lambda r: -r["mean_arbitrage"])
    raw = np.array([r["mean_arbitrage"] for r in rows])
    ren = np.array([r["mean_arbitrage_after_renorm"] for r in rows])
    labels = [short(r["model"]) for r in rows]
    o = fb["overall"]
    title = (
        f"Guaranteed arbitrage per unit of stake: {o['mean_arbitrage']:.2f} on average, "
        f"{o['mean_arbitrage_after_renorm']:.2f} once the four cells are renormalised"
    )

    def build(p: dict[str, Any]):
        fig, ax = plt.subplots(figsize=(8.6, 0.2 * len(rows) + 1.9))
        y = np.arange(len(rows))
        ax.hlines(y, ren, raw, color=p["axis"], linewidth=1.6, zorder=1)
        ax.scatter(raw, y, s=34, color=p["series"][0], zorder=3, label="as elicited")
        ax.scatter(
            ren, y, s=34, color=p["series"][1], zorder=3, label="after renormalising the four cells"
        )
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_ylim(-0.8, len(rows) - 0.2)
        ax.set_xlim(0, max(raw.max(), ren.max()) * 1.06)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(
            "guaranteed profit per unit of total stake (mean over a configuration's tuples)"
        )
        ax.legend(loc="upper right", fontsize=9, handletextpad=0.3)
        ax.set_title(title, loc="left", fontsize=10.5, pad=12)
        fig.tight_layout()
        return fig

    return build


def fig_brier(res: dict[str, Any]):
    fb = res["forecastbench"]
    rows = [
        r for r in fb["models"] if "brier_raw" in r and r["share_independence_constructed"] < 0.9
    ]
    assert all(r["brier_projected"] <= r["brier_raw"] for r in rows)
    x = np.array([r["brier_raw"] for r in rows])
    y = np.array([r["brier_projected"] for r in rows])
    o = fb["overall"]
    ci = o["ci95"]
    title = (
        f"Coherentization lowers the Brier score from {o['brier_raw']:.3f} to "
        f"{o['brier_projected']:.3f}, more than renormalising ({o['brier_renormalised']:.3f})"
    )

    def build(p: dict[str, Any]):
        fig, (a1, a2) = plt.subplots(
            1, 2, figsize=(10.2, 4.3), gridspec_kw={"width_ratios": [1, 1.15]}
        )
        lim = (0.1, max(x.max(), y.max()) * 1.04)
        a1.plot(lim, lim, color=p["axis"], linewidth=1.2, zorder=1)
        a1.scatter(x, y, s=14, color=p["series"][0], alpha=0.75, linewidths=0, zorder=2)
        a1.set_xlim(lim)
        a1.set_ylim(lim)
        a1.set_aspect("equal")
        a1.set_xlabel("Brier score as elicited")
        a1.set_ylabel("Brier score after Brier projection")
        a1.text(
            lim[1] * 0.99, lim[1] * 0.93, "no change", ha="right", fontsize=8.5, color=p["muted"]
        )
        a1.set_title(
            f"None of {len(rows)} model configurations gets worse", loc="left", fontsize=10
        )
        names = ["as elicited", "renormalised", "projected"]
        whole = [o["brier_raw"], o["brier_renormalised"], o["brier_projected"]]
        cells = [o["brier_cells_raw"], o["brier_cells_renormalised"], o["brier_cells_projected"]]
        err = [
            ci["brier_raw"]["ci95"],
            ci["brier_renormalised"]["ci95"],
            ci["brier_projected"]["ci95"],
        ]
        pos = np.arange(3)
        h = 0.36
        cols = [p["muted"], p["series"][1], p["series"][0]]
        a2.barh(pos - h / 2 - 0.02, whole, height=h, color=cols, zorder=2)
        a2.barh(pos + h / 2 + 0.02, cells, height=h, color=cols, alpha=0.5, zorder=2)
        for i in range(3):
            a2.errorbar(
                whole[i],
                pos[i] - h / 2 - 0.02,
                xerr=[[whole[i] - err[i][0]], [err[i][1] - whole[i]]],
                color=p["ink"],
                capsize=2,
                linewidth=1,
                zorder=3,
            )
            a2.text(
                whole[i] + 0.004,
                pos[i] - h / 2 - 0.02,
                f"{whole[i]:.3f}",
                va="center",
                fontsize=8.5,
                color=p["ink_secondary"],
            )
            a2.text(
                cells[i] + 0.004,
                pos[i] + h / 2 + 0.02,
                f"{cells[i]:.3f}",
                va="center",
                fontsize=8.5,
                color=p["ink_secondary"],
            )
        a2.set_yticks(pos, names)
        a2.invert_yaxis()
        a2.set_xlim(0, 0.27)
        a2.grid(axis="y", visible=False)
        a2.set_xlabel("pooled Brier score on resolved tuples (mean squared error)")
        a2.set_title(
            "Solid: all six forecasts. Faded: the four joint cells only", loc="left", fontsize=10
        )
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return fig

    return build


def fig_distribution(res: dict[str, Any]):
    fb = res["forecastbench"]
    e = fb["ecdf"]
    g = np.array(e["grid"])
    raw, ren = np.array(e["raw"]), np.array(e["renormalised"])
    o = fb["overall"]
    xmax = 0.6
    title = (
        f"{100 * o['share_arbitrage_ge_5pct']:.0f}% of tuples admit at least 5% profit per unit "
        "stake; "
        f"{100 * o['share_arbitrage_ge_5pct_after_renorm']:.0f}% still do after renormalising"
    )

    def build(p: dict[str, Any]):
        fig, ax = plt.subplots(figsize=(8.2, 4.4))
        sel = g <= xmax
        ax.plot(g[sel], raw[sel], color=p["series"][0], linewidth=2, label="as elicited")
        ax.plot(
            g[sel],
            ren[sel],
            color=p["series"][1],
            linewidth=2,
            label="after renormalising the four cells",
        )
        ax.axvline(0.05, color=p["muted"], linewidth=1, linestyle=(0, (4, 3)))
        ax.text(0.054, 0.04, "5% profit", fontsize=8.5, color=p["muted"])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("guaranteed profit per unit of total stake")
        ax.set_ylabel("share of tuples with at most this profit")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.5, pad=10)
        fig.tight_layout()
        return fig

    return build


def fig_geometry():
    def build(p: dict[str, Any]):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        q = np.array([0.9, 0.9])
        v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])  # worlds: A true, A false
        proj = np.array([0.5, 0.5])
        r1, r2 = np.linalg.norm(q - v1), np.linalg.norm(q - v2)
        t = np.linspace(0, 2 * np.pi, 400)
        # lens: points strictly closer to both worlds than q is (Joyce's dominance region)
        pts = np.stack([v1[0] + r1 * np.cos(t), v1[1] + r1 * np.sin(t)], axis=1)
        inside = np.linalg.norm(pts - v2, axis=1) <= r2
        grid = np.linspace(-0.2, 1.3, 600)
        gx, gy = np.meshgrid(grid, grid)
        lens = (np.hypot(gx - v1[0], gy - v1[1]) <= r1) & (np.hypot(gx - v2[0], gy - v2[1]) <= r2)
        ax.contourf(
            gx, gy, lens.astype(float), levels=[0.5, 1.5], colors=[p["series"][0]], alpha=0.16
        )
        for v, r in ((v1, r1), (v2, r2)):
            ax.add_patch(
                Circle(v, r, fill=False, edgecolor=p["axis"], linewidth=1.2, linestyle=(0, (5, 3)))
            )
        ax.add_patch(
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], fill=False, edgecolor=p["grid"], linewidth=1)
        )
        ax.plot(
            *zip(v1, v2, strict=True), color=p["series"][0], linewidth=2.4, solid_capstyle="round"
        )
        ax.plot([q[0], proj[0]], [q[1], proj[1]], color=p["series"][1], linewidth=1.6)
        for pt, c in ((v1, p["ink"]), (v2, p["ink"])):
            ax.scatter(*pt, s=46, color=c, zorder=4)
        ax.scatter(*q, s=60, color=p["series"][1], zorder=5)
        ax.scatter(*proj, s=60, color=p["series"][0], zorder=5)
        ax.annotate("A true", v1, (1.02, -0.07), fontsize=9, color=p["ink_secondary"], ha="center")
        ax.annotate("A false", v2, (-0.04, 1.06), fontsize=9, color=p["ink_secondary"], ha="center")
        ax.annotate(
            "forecast q = (0.9, 0.9)",
            q,
            (1.0, 1.07),
            fontsize=9,
            color=p["ink"],
            arrowprops={"arrowstyle": "-", "color": p["muted"], "linewidth": 0.8},
        )
        ax.annotate(
            "projection p",
            proj,
            (0.12, 0.45),
            fontsize=9,
            color=p["ink"],
            arrowprops={"arrowstyle": "-", "color": p["muted"], "linewidth": 0.8},
        )
        ax.text(
            0.12,
            -0.15,
            "coherent forecasts:\nP(A) + P(not A) = 1",
            fontsize=8.5,
            color=p["series"][0],
            rotation=0,
        )
        ax.set_xlim(-0.2, 1.45)
        ax.set_ylim(-0.2, 1.3)
        ax.set_aspect("equal")
        ax.set_xlabel("forecast of A")
        ax.set_ylabel("forecast of not A")
        ax.set_title(
            "Every coherent point in the lens beats q in both worlds", loc="left", fontsize=10.5
        )
        del inside
        fig.tight_layout()
        return fig

    return build


def main() -> None:
    res = json.loads(RESULTS.read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("incoherence-by-model", fig_incoherence_by_model(res)),
        ("brier-before-after", fig_brier(res)),
        ("arbitrage-distribution", fig_distribution(res)),
        ("geometry-of-dominance", fig_geometry()),
    ):
        for path in save_figure(build, FIGURES / name):
            log(f"wrote {path.relative_to(FIGURES.parent.parent)}")


if __name__ == "__main__":
    main()

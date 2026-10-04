"""Explanatory animation from docs/results.json: moving every forecast toward its projection."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from _common import RESULTS, ROOT, log
from animate import save_animation

from dutchbook.viz import style

N_FRAMES = 44


def main() -> None:
    path = json.loads(RESULTS.read_text())["forecastbench"]["repair_path"]
    t = np.array(path["t"])
    whole, cells = np.array(path["brier"]), np.array(path["brier_cells"])
    worse = path["tuples_worse_than_unrepaired"]
    n = path["n_tuples"]
    grid = np.linspace(0.0, 1.0, N_FRAMES)
    lo, hi = min(whole.min(), cells.min()), max(whole.max(), cells.max())

    def draw(ax: Any, p: dict[str, Any], i: int) -> None:
        s = grid[i]
        mask = t <= s + 1e-9
        ts = np.append(t[mask], s)
        w_now, c_now = np.interp(s, t, whole), np.interp(s, t, cells)
        ax.plot(ts, np.append(whole[mask], w_now), color=p["series"][0], linewidth=2.4)
        ax.plot(ts, np.append(cells[mask], c_now), color=p["series"][1], linewidth=2.4)
        ax.scatter([s, s], [w_now, c_now], color=[p["series"][0], p["series"][1]], s=46, zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(lo - 0.012, hi + 0.012)
        ax.set_xlabel("share of the way from each forecast to its coherent projection")
        ax.set_ylabel("pooled Brier score")
        ax.figure.subplots_adjust(left=0.11, bottom=0.17, right=0.97, top=0.9)
        ax.text(0.5, 0.1795, "all six forecasts", color=p["series"][0], fontsize=9.5)
        ax.text(0.42, 0.1565, "four joint cells only", color=p["series"][1], fontsize=9.5)
        k = round(s * (len(t) - 1))
        ax.text(
            0.98,
            0.95,
            f"{worse[k]} of {n:,} resolved tuples\nscore worse than unrepaired",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=p["ink_secondary"],
        )
        ax.set_title(
            f"Repairing incoherent forecasts lowers the Brier score: {w_now:.3f} at {s:.0%}",
            loc="left",
            fontsize=10.5,
        )

    for out in save_animation(
        draw,
        N_FRAMES,
        ROOT / "docs" / "animations" / "repair-path",
        style=style,
        figsize=(7.4, 4.0),
        fps=16,
        dpi=100,
        hold_last=28,
    ):
        log(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

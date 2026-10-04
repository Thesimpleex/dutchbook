"""Build the self-contained HTML audit report of the ForecastBench experiment."""

from __future__ import annotations

import numpy as np
from _common import FIGURES, ROOT, load_results, log

from dutchbook import audit
from dutchbook.forecastbench import FORMULAS
from dutchbook.logic import parse
from dutchbook.report import forecastbench_report_html

OUT = ROOT / "docs" / "forecastbench-report.html"


def main() -> None:
    res = load_results()["forecastbench"]
    rows = [
        r
        for r in res["models"]
        if r["variant"] == "zero shot"
        and r["n_tuples"] >= 1000
        and r["share_independence_constructed"] < 0.9
    ]
    hero = res["hero_tuple"]
    q = np.array(list(hero["forecasts"].values()))
    ex = audit([parse(f) for f in FORMULAS], q, labels=[str(parse(f)) for f in FORMULAS])
    html = forecastbench_report_html(
        overall=res["overall"],
        rows=rows,
        snapshot=res["snapshot"],
        example=ex,
        example_text={"A": hero["questions"][0], "B": hero["questions"][1]},
        example_caption=(
            f"{hero['model']}, round {hero['round']}: the tuple with the median guaranteed "
            "arbitrage among the model's resolved market questions."
        ),
        figures=[
            (
                "Mean guaranteed arbitrage per configuration, as elicited and after renormalising.",
                FIGURES / "incoherence-by-model-light.png",
            ),
            (
                "Brier score before and after the Brier projection, and against renormalising.",
                FIGURES / "brier-before-after-light.png",
            ),
        ],
    )
    OUT.write_text(html, encoding="utf-8")
    log(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

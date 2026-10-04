"""Reproduce every number, figure, animation and screenshot of the README.

Downloads the ForecastBench snapshot into ``data/raw/`` (about 133 MB, skipped when the
local copy matches the remote ETag), audits every combination tuple, and rewrites
``docs/results.json``, ``docs/figures/``, ``docs/animations/``, ``docs/screenshots/``,
``docs/forecastbench-report.html`` and the files under ``examples/``.

Run from the repository root after ``pip install -e ".[dev,docs]"``::

    python scripts/reproduce_all.py [--workers N] [--skip STEP ...]

A cold run took about 10 minutes on a ten-core laptop (the exact Dutch-book LP of every
tuple is the bulk of it); later runs reuse the caches under ``data/cache/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STEPS = (
    ("validation", "validation_experiments.py", False),
    ("forecastbench", "forecastbench_experiment.py", True),
    ("pairwise", "pairwise_experiment.py", True),
    ("figures", "make_figures.py", False),
    ("animations", "make_animations.py", False),
    ("report", "make_report.py", False),
    ("screenshots", "make_screenshots.py", False),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, help="processes for the LP stages (default: all cores)")
    ap.add_argument("--skip", nargs="*", default=[], choices=[s[0] for s in STEPS])
    args = ap.parse_args()
    for name, script, parallel in STEPS:
        if name in args.skip:
            continue
        cmd = [sys.executable, str(SCRIPTS / script)]
        if parallel and args.workers:
            cmd += ["--workers", str(args.workers)]
        start = time.time()
        print(f"[{name}] {script}", flush=True)
        subprocess.run(cmd, check=True, cwd=SCRIPTS)
        print(f"[{name}] done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()

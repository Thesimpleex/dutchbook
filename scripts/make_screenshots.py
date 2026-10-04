"""Regenerate the product screenshots from the real CLI and the real HTML report."""

from __future__ import annotations

import sys
from pathlib import Path

from _common import ROOT, SCREENSHOTS, log
from animate import terminal_cast_svg
from screenshots import report_screenshot, terminal_screenshot

CLI = str(ROOT / ".venv" / "bin" / "dutchbook")
SPEC = "examples/forecastbench_tuple.toml"
DATA = "examples/forecastbench_tuple.csv"
REPORT = ROOT / "docs" / "forecastbench-report.html"


def terminal(name: str, args: list[str], display: str, title: str, columns: int) -> None:
    command = [CLI if Path(CLI).exists() else "dutchbook", *args]
    for theme in ("dark", "light"):
        out = SCREENSHOTS / f"{name}-{theme}.png"
        terminal_screenshot(
            command,
            out,
            title=title,
            display_command=display,
            theme=theme,
            columns=columns,
            cwd=ROOT,
            allow_failure=True,
        )
        log(f"wrote {out.relative_to(ROOT)}")


def report(name: str, selector: str | None, clip: int | None, width: int = 1100) -> None:
    for theme in ("dark", "light"):
        out = SCREENSHOTS / f"{name}-{theme}.png"
        report_screenshot(
            REPORT, out, width=width, selector=selector, clip_height=clip, color_scheme=theme
        )
        log(f"wrote {out.relative_to(ROOT)}")


def cast() -> None:
    out = ROOT / "docs" / "animations" / "audit.svg"
    terminal_cast_svg(
        [CLI if Path(CLI).exists() else "dutchbook", "audit", DATA, "--spec", SPEC],
        out,
        title="dutchbook audit",
        display_command=f"dutchbook audit {DATA} --spec {SPEC}",
        columns=108,
        cwd=ROOT,
        allow_failure=True,
    )
    log(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e3:.0f} kB)")


def main() -> None:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    cast()
    terminal(
        "audit",
        ["audit", DATA, "--spec", SPEC],
        f"dutchbook audit {DATA} --spec {SPEC}",
        "dutchbook audit",
        columns=108,
    )
    terminal(
        "repair",
        ["repair", DATA, "--spec", SPEC],
        f"dutchbook repair {DATA} --spec {SPEC}",
        "dutchbook repair",
        columns=108,
    )
    terminal(
        "leaderboard",
        ["forecastbench", "--round", "2024-07-21", "--no-download", "--top", "12"],
        "dutchbook forecastbench --round 2024-07-21",
        "dutchbook forecastbench",
        columns=124,
    )
    if REPORT.exists():
        report("report-overview", None, 1000)
        report("report-example", "#example", None)
    else:
        log("docs/forecastbench-report.html missing: run scripts/make_report.py first")
        sys.exit(1)


if __name__ == "__main__":
    main()

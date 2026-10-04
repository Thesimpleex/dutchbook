"""Command line interface: ``dutchbook audit | repair | bounds | forecastbench``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
from rich.table import Table
from rich.text import Text

from dutchbook import __version__
from dutchbook.audit import Audit, Method, audit, repair
from dutchbook.coherence import STAKE_NORMS, StakeNorm
from dutchbook.extension import IncoherentForecastsError, extension_bounds
from dutchbook.render import (
    header_line,
    make_console,
    render_audit,
    render_bounds,
    render_repair,
)
from dutchbook.report import audit_report_html
from dutchbook.spec import ForecastSet, Spec, load_spec, read_forecasts

_NORM_TEXT = {
    "l1": "total stake at most 1",
    "linf": "each stake at most 1",
    "l2": "Euclidean stake norm at most 1",
}


def _add_common(p: argparse.ArgumentParser, method: bool = False) -> None:
    p.add_argument("forecasts", type=Path, help="CSV with columns id, forecast [, group]")
    p.add_argument("--spec", type=Path, required=True, help="TOML file defining the formulas")
    p.add_argument("--group", help="only this forecast set (default: all, detail for the worst)")
    p.add_argument("--width", type=int, help="console width (default: terminal width)")
    if method:
        p.add_argument(
            "--method",
            choices=("auto", "enumerate", "column-generation"),
            default="auto",
            help="list all worlds, or generate them with a MILP oracle (auto: beyond 16 atoms)",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dutchbook",
        description="Find the Dutch book in probability forecasts over logically related events, "
        "price it exactly, and repair the forecasts.",
    )
    parser.add_argument("--version", action="version", version=f"dutchbook {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="test forecasts for coherence and price the Dutch book")
    _add_common(a, method=True)
    a.add_argument(
        "--stake-norm", choices=STAKE_NORMS, default="l1", help="norm bounding the stakes"
    )
    a.add_argument(
        "--stake",
        type=float,
        default=1.0,
        help="bound on the stake norm; scales stakes and profit linearly (default 1)",
    )
    a.add_argument("--html", type=Path, help="write a self-contained HTML report")
    a.add_argument("--json", type=Path, help="write the result as JSON")

    r = sub.add_parser("repair", help="coherentize forecasts and certify the improvement")
    _add_common(r, method=True)
    r.add_argument("--divergence", choices=("brier", "kl"), default="brier")
    r.add_argument("--out", type=Path, help="write repaired forecasts as CSV (group,id,forecast)")
    r.add_argument("--certificate", type=Path, help="write the dominance certificate as JSON")

    b = sub.add_parser("bounds", help="coherent interval for a further formula")
    _add_common(b)
    b.add_argument("--formula", required=True, help="the formula to bound, e.g. 'B'")

    f = sub.add_parser(
        "forecastbench", help="audit ForecastBench combination questions (downloads the data)"
    )
    f.add_argument("--round", dest="round_date", help="forecast due date, e.g. 2024-07-21")
    f.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    f.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    f.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    f.add_argument("--workers", type=int, help="processes for the LPs (default: all cores)")
    f.add_argument(
        "--variant", default="zero shot", help="prompt variant to list (default: zero shot)"
    )
    f.add_argument("--all-variants", action="store_true", help="list every configuration")
    f.add_argument("--min-tuples", type=int, default=500, help="smallest configuration to list")
    f.add_argument("--top", type=int, default=15, help="rows to print (most coherent first)")
    f.add_argument("--html", type=Path, help="write the HTML audit report with the leaderboard")
    f.add_argument("--width", type=int, help="console width")
    return parser


def _select(sets: list[ForecastSet], group: str | None) -> list[ForecastSet]:
    if group is None:
        return sets
    chosen = [s for s in sets if s.group == group]
    if not chosen:
        raise ValueError(f"no forecast set named {group!r}; available: {[s.group for s in sets]}")
    return chosen


def _audit_one(spec: Spec, fs: ForecastSet, stake_norm: StakeNorm, method: Method) -> Audit:
    formulas = [spec.formula(i) for i in fs.ids]
    return audit(
        formulas,
        fs.values,
        labels=_pretty_labels(spec, fs.ids),
        stake_norm=stake_norm,
        method=method,
    )


def _pretty_labels(spec: Spec, ids: Sequence[str]) -> list[str]:
    return [str(spec.formula(i)) for i in ids]


def _audit_json(
    name: str, a: Audit, spec: Spec, ids: Sequence[str], stake_norm: str, scale: float
) -> dict[str, Any]:
    return {
        "group": name,
        "coherent": a.coherent,
        "stake_norm": stake_norm,
        "guaranteed_profit": a.book.profit * scale,
        "distances": a.distances,
        "forecasts": {i: float(v) for i, v in zip(ids, a.forecasts, strict=True)},
        "orders": [
            {
                "id": ids[o.index],
                "formula": str(spec.formula(ids[o.index])),
                "side": o.side,
                "price": o.price,
                "stake": o.stake,
            }
            for o in a.book.orders(a.forecasts, list(ids), scale)
        ],
        "payoffs": [
            {"world": a.space.describe_world(w), "net_payoff": float(a.book.payoffs[w] * scale)}
            for w in range(a.space.W.shape[0])
        ],
    }


def cmd_audit(args: argparse.Namespace, console: Console) -> int:
    spec = load_spec(args.spec)
    sets = _select(read_forecasts(args.forecasts, spec), args.group)
    results = [(s, _audit_one(spec, s, args.stake_norm, args.method)) for s in sets]
    first = results[0][1]
    info = (
        [f"{len(first.labels)} forecasts", f"{len(first.space.atoms)} atoms"]
        + ([f"{first.space.W.shape[0]} worlds"] if first.exhaustive else [])
        if len(results) == 1
        else [f"{len(results)} forecast sets"]
    )
    header_line(console, "dutchbook audit", [args.forecasts.name, *info])
    worst_set, worst = max(results, key=lambda r: r[1].book.profit)
    if len(results) > 1:
        console.print()
        console.print(_summary_table(results, args.stake))
    shown = worst_set.group if len(results) > 1 else None
    render_audit(
        console,
        worst,
        args.stake_norm,
        scale=args.stake,
        descriptions=spec.atoms,
        group=shown,
    )
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(
            audit_report_html(
                [(s.group, a) for s, a in results],
                title=spec.title or "Coherence audit",
                source=args.forecasts.name,
                descriptions=spec.atoms,
                stake_norm_text=_NORM_TEXT[args.stake_norm],
                scale=args.stake,
            ),
            encoding="utf-8",
        )
        console.print(f"\nHTML report written to {args.html}", style="dim")
    if args.json:
        payload = {
            "source": args.forecasts.name,
            "stake_norm": args.stake_norm,
            "scale": args.stake,
            "groups": [
                _audit_json(s.group, a, spec, s.ids, args.stake_norm, args.stake)
                for s, a in results
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 1 if any(not a.coherent for _, a in results) else 0


def _summary_table(results: Sequence[tuple[ForecastSet, Audit]], scale: float) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("Forecast set")
    table.add_column("Forecasts", justify="right")
    table.add_column("Status")
    table.add_column("Guaranteed profit", justify="right")
    table.add_column("Euclidean distance", justify="right")
    for s, a in sorted(results, key=lambda r: -r[1].book.profit):
        status = Text("coherent", "green") if a.coherent else Text("incoherent", "red")
        table.add_row(
            s.group,
            str(len(s.ids)),
            status,
            f"{a.book.profit * scale:.3f}",
            f"{a.distances['l2']:.3f}",
        )
    return table


def cmd_repair(args: argparse.Namespace, console: Console) -> int:
    spec = load_spec(args.spec)
    sets = _select(read_forecasts(args.forecasts, spec), args.group)
    repairs = []
    for s in sets:
        formulas = [spec.formula(i) for i in s.ids]
        rep = repair(
            formulas,
            s.values,
            labels=_pretty_labels(spec, s.ids),
            divergence=args.divergence,
            method=args.method,
        )
        repairs.append((s, rep))
    first_rep = repairs[0][1]
    info = (
        [f"{len(first_rep.labels)} forecasts", f"{first_rep.space.W.shape[0]} worlds"]
        if len(repairs) == 1
        else [f"{len(repairs)} forecast sets"]
    )
    kind = "Brier projection" if args.divergence == "brier" else "KL projection"
    header_line(console, "dutchbook repair", [args.forecasts.name, kind, *info])
    worst_set, worst = max(repairs, key=lambda r: r[1].certificate.divergence)
    if len(repairs) > 1:
        table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
        table.add_column("Forecast set")
        table.add_column("D(p, q)", justify="right")
        table.add_column("Min improvement", justify="right")
        table.add_column("Dominance")
        for s, rep in sorted(repairs, key=lambda r: -r[1].certificate.divergence):
            c = rep.certificate
            ok = Text("holds", "green") if c.holds else Text("VIOLATED", "bold red")
            table.add_row(s.group, f"{c.divergence:.4f}", f"{c.min_improvement:.4f}", ok)
        console.print()
        console.print(table)
    render_repair(
        console,
        worst,
        descriptions=spec.atoms,
        group=worst_set.group if len(repairs) > 1 else None,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["group", "id", "forecast"])
            for s, rep in repairs:
                for ident, value in zip(s.ids, rep.repaired, strict=True):
                    writer.writerow([s.group, ident, f"{value:.10g}"])
        console.print(f"\nRepaired forecasts written to {args.out}", style="dim")
    if args.certificate:
        payload = [
            {
                "group": s.group,
                "divergence": args.divergence,
                "scoring": rep.certificate.scoring,
                "guaranteed_improvement": rep.certificate.divergence,
                "min_improvement": rep.certificate.min_improvement,
                "duality_gap": rep.certificate.gap,
                "holds": rep.certificate.holds,
                "worlds": [
                    {
                        "world": rep.space.describe_world(w),
                        "score_before": float(rep.certificate.score_before[w]),
                        "score_after": float(rep.certificate.score_after[w]),
                        "improvement": float(rep.certificate.improvement[w]),
                    }
                    for w in range(rep.space.W.shape[0])
                ],
            }
            for s, rep in repairs
        ]
        args.certificate.parent.mkdir(parents=True, exist_ok=True)
        args.certificate.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_bounds(args: argparse.Namespace, console: Console) -> int:
    spec = load_spec(args.spec)
    sets = _select(read_forecasts(args.forecasts, spec), args.group)
    if len(sets) > 1:
        raise ValueError("several forecast sets found; choose one with --group")
    s = sets[0]
    formulas = [spec.formula(i) for i in s.ids]
    try:
        bounds = extension_bounds(formulas, s.values, args.formula)
    except IncoherentForecastsError as exc:
        console.print(Text(f"INCOHERENT  {exc}", style="bold red"))
        return 1
    header_line(console, "dutchbook bounds", [args.forecasts.name, f"{len(s.ids)} forecasts"])
    render_bounds(console, args.formula, bounds, s.ids)
    return 0


def _status(console: Console, message: str) -> AbstractContextManager[Any]:
    """A spinner on interactive terminals, nothing when output is piped."""
    return console.status(message) if sys.stdout.isatty() else nullcontext()


def cmd_forecastbench(args: argparse.Namespace, console: Console) -> int:
    from dutchbook import benchmark, forecastbench
    from dutchbook.report import forecastbench_report_html

    raw = forecastbench.DEFAULT_RAW_DIR if args.raw_dir is None else args.raw_dir
    target = raw / forecastbench.SNAPSHOT_NAME
    if args.no_download and not target.exists():
        raise FileNotFoundError(f"{target} not found (--no-download given)")
    if not args.no_download:
        with Progress(
            TextColumn("downloading ForecastBench snapshot"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
            transient=True,
        ) as bar:
            task = bar.add_task("dl", total=None)
            manifest = forecastbench.download_snapshot(
                raw, progress=lambda done, total: bar.update(task, completed=done, total=total)
            )
    else:
        manifest = forecastbench.read_manifest(raw)
    with _status(console, "reading combination tuples"):
        table = forecastbench.load_tuples(raw, args.cache_dir)
    rounds = list(table.rounds)
    selected = rounds if args.round_date is None else [args.round_date]
    for r in selected:
        if r not in rounds:
            raise ValueError(f"round {r!r} has no combination questions; available: {rounds}")
    with _status(console, "computing exact Dutch books and projections (cached per round)"):
        metrics = benchmark.cached_metrics(
            table, selected, args.cache_dir, forecastbench.snapshot_digest(raw), args.workers
        )
    mask = np.isin(table.round_idx, [rounds.index(r) for r in selected])
    sub = table.select(mask)
    sub_metrics = metrics.select(mask)
    rows = benchmark.model_rows(sub, sub_metrics)
    listed = _listed_rows(rows, args)
    overall = benchmark.overall_summary(sub, sub_metrics)
    round_label = selected[0] if len(selected) == 1 else f"{len(selected)} rounds"
    header_line(
        console,
        "dutchbook forecastbench",
        [
            f"round {round_label}",
            f"snapshot {manifest.get('downloaded', '')}",
            f"{overall['n_tuples']:,} tuples",
            f"{overall['n_models']} configurations",
        ],
    )
    console.print()
    console.print(
        Text.assemble(
            ("Caveat  ", "bold yellow"),
            "each joint cell was elicited in a separate prompt, so part of the incoherence is an "
            "elicitation artefact; the renormalised column rescales the four cells to sum to one.",
        ),
        width=min(console.width, 100),
    )
    console.print()
    console.print(_leaderboard_table(listed[: args.top]))
    scope = "all variants" if args.all_variants else args.variant
    shown = min(len(listed), args.top)
    console.print(
        f"{shown} of {len(listed)} configurations shown ({scope}, at least {args.min_tuples} "
        "tuples, most coherent first)\nprofit: guaranteed gain per unit of total stake",
        style="dim",
    )
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(forecastbench_report_html(overall, listed, manifest), encoding="utf-8")
        console.print(f"\nHTML report written to {args.html}", style="dim")
    return 0


def _listed_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    keep = [
        r
        for r in rows
        if r["kind"] == "model"
        and r["n_tuples"] >= args.min_tuples
        and r["share_independence_constructed"] < 0.9
        and (args.all_variants or r["variant"] == args.variant)
    ]
    return sorted(keep, key=lambda r: r["mean_arbitrage"])


def _leaderboard_table(rows: Sequence[dict[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("Model configuration", no_wrap=True, overflow="ellipsis", max_width=52)
    table.add_column("Tuples", justify="right")
    table.add_column("Profit ≥ 5%", justify="right")
    table.add_column("Mean profit", justify="right")
    table.add_column("After renorm.", justify="right")
    table.add_column("Brier before", justify="right")
    table.add_column("after", justify="right")
    for r in rows:
        has = "brier_raw" in r
        table.add_row(
            str(r["model"]),
            f"{int(r['n_tuples']):,}",
            f"{100 * r['share_arbitrage_ge_5pct']:.0f}%",
            Text(f"{r['mean_arbitrage']:.3f}", style="bold"),
            f"{r['mean_arbitrage_after_renorm']:.3f}",
            f"{r['brier_raw']:.3f}" if has else "n/a",
            Text(f"{r['brier_projected']:.3f}", style="green") if has else "n/a",
        )
    return table


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = make_console(getattr(args, "width", None))
    handlers = {
        "audit": cmd_audit,
        "repair": cmd_repair,
        "bounds": cmd_bounds,
        "forecastbench": cmd_forecastbench,
    }
    try:
        return handlers[args.command](args, console)
    except (ValueError, FileNotFoundError, KeyError, OSError, RuntimeError) as exc:
        err = Console(stderr=True, highlight=False)
        err.print(f"dutchbook: error: {exc}", style="red", markup=False)
        return 2


if __name__ == "__main__":
    sys.exit(main())

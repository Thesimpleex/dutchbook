"""Self-contained HTML reports (inline CSS, no scripts, no external assets)."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from dutchbook.audit import Audit
from dutchbook.render import price

_CSS = """
:root { --bg:#fff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --surface:#fcfcfb; --blue:#2a78d6; --orange:#eb6834; --green:#007a00; --red:#c93a39;
  --amber:#9a6a00; --amber-bg:#fdf6e3; --amber-line:#ecd9a0; --amber-ink:#4a4331;
  --neutral:#f0efec; --bad-bg:#fbe3e2; --ok-bg:#dff3df; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --ink:#f4f3ee; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
    --surface:#1a1a19; --blue:#3987e5; --orange:#d95926; --green:#3cc993; --red:#f08c8b;
    --amber:#f5b83a; --amber-bg:#26220f; --amber-line:#4a4120; --amber-ink:#e6dcb8;
    --neutral:#2c2c2a; --bad-bg:#3a1f1f; --ok-bg:#16301f; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); line-height:1.5;
  font:15px/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif; -webkit-font-smoothing:antialiased; }
main { max-width:980px; margin:0 auto; padding:56px 40px 72px; }
header { border-bottom:1px solid var(--grid); padding-bottom:28px; margin-bottom:36px; }
.kicker { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
h1 { font-size:30px; line-height:1.2; margin:8px 0 10px; font-weight:600; letter-spacing:-.01em; }
h2 { font-size:18px; margin:44px 0 14px; font-weight:600; }
p { margin:0 0 12px; color:var(--ink2); }
.lede { font-size:16px; color:var(--ink2); max-width:760px; }
.meta { font-size:13px; color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin:28px 0 8px; }
.card { border:1px solid var(--grid); border-radius:8px; padding:16px 18px; background:var(--surface); }
.card .v { font-size:24px; white-space:nowrap; font-weight:600; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
.card .l { font-size:12.5px; color:var(--ink2); margin-top:2px; }
.card .s { font-size:12px; color:var(--muted); margin-top:6px; }
.callout { border:1px solid var(--amber-line); background:var(--amber-bg); border-radius:8px;
  padding:16px 20px; margin:28px 0; }
.callout b { color:var(--amber); }
.callout p { color:var(--amber-ink); margin:6px 0 0; font-size:14px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13.5px; }
th { text-align:left; font-weight:600; font-size:12px; color:var(--ink2); padding:8px 10px;
  border-bottom:1px solid var(--ink); white-space:nowrap; }
td { padding:7px 10px; border-bottom:1px solid var(--grid); vertical-align:middle;
  white-space:nowrap; }
th.n, td.n { text-align:right; }
td.dim { color:var(--muted); }
tr.total td { border-top:1px solid var(--ink); border-bottom:none; font-weight:600; }
.pos { color:var(--green); font-weight:600; } .neg { color:var(--red); }
.chip { display:inline-block; padding:1px 9px; border-radius:10px; font-size:12px; font-weight:600; }
.chip.bad { background:var(--bad-bg); color:var(--red); }
.chip.ok { background:var(--ok-bg); color:var(--green); }
.bars { width:120px; } .bar { height:7px; border-radius:2px; background:var(--neutral);
  position:relative; margin:2px 0; }
.bar span { position:absolute; left:0; top:0; bottom:0; border-radius:2px; }
.bar .raw { background:var(--blue); } .bar .base { background:var(--orange); }
.legend { font-size:12px; color:var(--ink2); margin:0 0 10px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 5px 0 12px;
  vertical-align:-1px; } .legend i:first-child { margin-left:0; }
figure { margin:20px 0; } figure img { max-width:100%; border:1px solid var(--grid); border-radius:6px; }
figcaption { font-size:12.5px; color:var(--ink2); margin-top:8px; }
.events { margin:0 0 18px; padding:0; list-style:none; font-size:14px; color:var(--ink2); }
.events b { color:var(--ink); display:inline-block; min-width:22px; }
footer { margin-top:56px; padding-top:18px; border-top:1px solid var(--grid); font-size:12.5px;
  color:var(--muted); }
code { font:12.5px ui-monospace, "SF Mono", Menlo, monospace; background:var(--neutral);
  padding:1px 5px; border-radius:3px; }
"""


def e(value: object) -> str:
    return escape(str(value))


def page(title: str, kicker: str, lede: str, body: str, footer: str = "") -> str:
    """Wrap a body in the report shell."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{_CSS}</style></head><body><main>"
        f'<header><div class="kicker">{e(kicker)}</div><h1>{e(title)}</h1>'
        f'<p class="lede">{lede}</p><div class="meta">Generated {stamp} by dutchbook</div></header>'
        f"{body}<footer>{footer}</footer></main></body></html>\n"
    )


def cards(items: Sequence[tuple[str, str, str]]) -> str:
    """``(value, label, note)`` triples as a row of cards."""
    inner = "".join(
        f'<div class="card"><div class="v">{e(v)}</div><div class="l">{e(label)}</div>'
        + (f'<div class="s">{e(note)}</div>' if note else "")
        + "</div>"
        for v, label, note in items
    )
    return f'<div class="cards">{inner}</div>'


def callout(title: str, text: str) -> str:
    return f'<div class="callout"><b>{e(title)}</b><p>{text}</p></div>'


def bar_pair(raw: float, base: float | None, scale: float) -> str:
    """Two thin horizontal bars (raw, optional baseline) on a common scale."""

    def one(cls: str, v: float) -> str:
        w = max(0.0, min(100.0, 100.0 * v / scale)) if scale > 0 else 0.0
        return f'<div class="bar"><span class="{cls}" style="width:{w:.1f}%"></span></div>'

    return (
        f'<div class="bars">{one("raw", raw)}{one("base", base) if base is not None else ""}</div>'
    )


def money(x: float, digits: int = 3) -> str:
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls}">{x:+.{digits}f}</span>'


def events_list(descriptions: Mapping[str, str], atoms: Sequence[str]) -> str:
    items = "".join(
        f"<li><b>{e(a)}</b> {e(descriptions[a])}</li>" for a in atoms if a in descriptions
    )
    return f'<ul class="events">{items}</ul>' if items else ""


def book_table(audit: Audit, scale: float = 1.0) -> str:
    """Bets and the payoff in every world, as an HTML table."""
    book, space = audit.book, audit.space
    orders = book.orders(audit.forecasts, audit.labels, scale)
    bets = "".join(
        f'<tr><td class="dim n">{k}</td><td><span class="{"pos" if o.side == "BUY" else "neg"}">'
        f'{o.side}</span></td><td>{e(o.label)}</td><td class="n">{price(o.price)}</td>'
        f'<td class="n">{o.stake:.3f}</td><td class="n">{-o.cost:+.3f}</td></tr>'
        for k, o in enumerate(orders, start=1)
    )
    bet_html = (
        '<table><thead><tr><th class="n">#</th><th>Side</th><th>Contract on</th>'
        '<th class="n">Price</th><th class="n">Stake</th><th class="n">Cash now</th></tr></thead>'
        f"<tbody>{bets}</tbody></table>"
    )
    show_bets = len(orders) <= 6
    head = "".join(f'<th class="n">#{k + 1}</th>' for k in range(len(orders))) if show_bets else ""
    rows = []
    for w in range(space.W.shape[0]):
        per = ""
        if show_bets:
            for o in orders:
                sign = 1.0 if o.side == "BUY" else -1.0
                v = sign * o.stake * (space.W[w, o.index] - audit.forecasts[o.index])
                per += f'<td class="n dim">{v:+.3f}</td>'
        rows.append(
            f"<tr><td>{e(space.describe_world(w))}</td>{per}"
            f'<td class="n">{money(book.payoffs[w] * scale)}</td></tr>'
        )
    cols = len(orders) if show_bets else 0
    total = (
        f'<tr class="total"><td>Guaranteed in every world</td>{"<td></td>" * cols}'
        f'<td class="n">{money(book.profit * scale)}</td></tr>'
    )
    world_html = (
        f'<table><thead><tr><th>World</th>{head}<th class="n">Net payoff</th></tr></thead>'
        f"<tbody>{''.join(rows)}{total}</tbody></table>"
    )
    return (
        "<h2>The Dutch book</h2><p>Bets placed at the forecaster's own prices. A positive stake "
        "buys a unit-payoff contract, a negative stake sells one.</p>"
        f"{bet_html}<h2>Net payoff in every world</h2>{world_html}"
    )


def audit_report_html(
    groups: Sequence[tuple[str, Audit]],
    title: str = "Coherence audit",
    source: str = "",
    descriptions: Mapping[str, str] | None = None,
    stake_norm_text: str = "total stake at most 1",
    scale: float = 1.0,
) -> str:
    """Report for one or more forecast sets: leaderboard plus the worst Dutch book."""
    ranked = sorted(groups, key=lambda g: (g[1].book.profit, g[0]))
    n_bad = sum(not a.coherent for _, a in groups)
    worst_name, worst = max(groups, key=lambda g: g[1].book.profit)
    scale_max = max((a.book.profit for _, a in groups), default=0.0) * scale or 1.0
    rows = "".join(
        f'<tr><td class="dim n">{k}</td><td>{e(name)}</td><td class="n">{len(a.labels)}</td>'
        f'<td><span class="chip {"ok" if a.coherent else "bad"}">'
        f"{'coherent' if a.coherent else 'incoherent'}</span></td>"
        f'<td class="n">{a.book.profit * scale:.3f}</td>'
        f"<td>{bar_pair(a.book.profit * scale, None, scale_max)}</td>"
        f'<td class="n">{a.distances["l2"]:.3f}</td></tr>'
        for k, (name, a) in enumerate(ranked, start=1)
    )
    board = (
        "<h2>Coherence leaderboard</h2><p>Most coherent first. The guaranteed profit is what a "
        f"bettor can lock in against the forecasts under the stake norm ({e(stake_norm_text)}); "
        'zero means coherent.</p><table><thead><tr><th class="n">Rank</th><th>Forecast set</th>'
        '<th class="n">Forecasts</th><th>Status</th><th class="n">Guaranteed profit</th><th></th>'
        '<th class="n">Euclidean distance</th></tr></thead>'
        f"<tbody>{rows}</tbody></table>"
    )
    body = cards(
        [
            (f"{n_bad} of {len(groups)}", "forecast sets are incoherent", ""),
            (f"{worst.book.profit * scale:.3f}", "largest guaranteed profit", worst_name),
            (
                f"{sum(a.book.profit for _, a in groups) / len(groups) * scale:.3f}",
                "mean guaranteed profit",
                stake_norm_text,
            ),
        ]
    )
    body += board
    if not worst.coherent:
        body += f"<h2>Worst case: {e(worst_name)}</h2>"
        if descriptions:
            body += events_list(descriptions, worst.space.atoms)
        body += book_table(worst, scale)
    lede = (
        "A Dutch book is a set of bets that loses money in every possible outcome. By "
        "de Finetti's theorem a set of probabilities over logically related events is safe from "
        "one exactly when it is coherent, that is, when it is a mixture of possible worlds."
    )
    footer = f"Source: <code>{e(source)}</code>. " if source else ""
    footer += "Computed by exact linear programming (HiGHS); no outcome data is used."
    return page(title, "dutchbook · audit", lede, body, footer)


def embed_png(path: str | Path, caption: str, alt: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return (
        f'<figure><img alt="{e(alt)}" src="data:image/png;base64,{data}">'
        f"<figcaption>{e(caption)}</figcaption></figure>"
    )


def forecastbench_report_html(
    overall: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    example: Audit | None = None,
    example_text: Mapping[str, str] | None = None,
    example_caption: str = "",
    figures: Sequence[tuple[str, str | Path]] = (),
    title: str = "ForecastBench combination questions: coherence audit",
) -> str:
    """Coherence leaderboard of language-model forecasters on ForecastBench combinations."""
    n = int(overall["n_tuples"])
    cards_html = cards(
        [
            (f"{n:,}", "tuples audited", f"{overall['n_models']} model configurations"),
            (
                f"{100 * overall['share_arbitrage_ge_5pct']:.0f}%",
                "of tuples admit at least 5% profit per unit stake",
                f"{100 * overall['share_dutch_bookable']:.1f}% admit any Dutch book",
            ),
            (
                f"{overall['mean_arbitrage']:.3f}",
                "mean guaranteed profit per unit of total stake",
                f"{overall['mean_arbitrage_after_renorm']:.3f} after renormalising the four cells",
            ),
            (
                f"{overall['brier_raw']:.3f} to {overall['brier_projected']:.3f}",
                "Brier score before and after coherentization",
                f"{int(overall['dominance_violations'])} violations of per-world dominance",
            ),
        ]
    )
    caveat = callout(
        "Read this first: part of the incoherence is an elicitation artefact.",
        "ForecastBench asked for each of the four joint outcomes in a separate prompt, so a model "
        "that forecasts every cell independently will not make them sum to one (the mean sum here "
        f"is {overall['mean_sum_joint_cells']:.2f}). The renormalised baseline rescales the four "
        "cells to sum to one; the marginal-versus-joint inconsistency that remains is the more "
        "robust finding, although the marginals were also elicited in their own prompts.",
    )
    scale = max(
        [float(r["mean_arbitrage"]) for r in rows]
        + [float(r["mean_arbitrage_after_renorm"]) for r in rows]
        or [1.0]
    )
    variants = {str(r.get("variant", "")) for r in rows}
    common = variants.pop() if len(variants) == 1 else ""
    suffix = f" ({common})" if common else ""
    body_rows = []
    for k, r in enumerate(rows, start=1):
        brier = f"{r['brier_raw']:.3f} to {r['brier_projected']:.3f}" if "brier_raw" in r else "n/a"
        body_rows.append(
            f'<tr><td class="dim n">{k}</td><td>{e(str(r["model"]).removesuffix(suffix))}</td>'
            f'<td class="n">{int(r["n_tuples"]):,}</td>'
            f'<td class="n">{100 * r["share_arbitrage_ge_5pct"]:.0f}%</td>'
            f'<td class="n">{r["mean_arbitrage"]:.3f}</td>'
            f'<td class="n">{r["mean_arbitrage_after_renorm"]:.3f}</td>'
            f"<td>{bar_pair(r['mean_arbitrage'], r['mean_arbitrage_after_renorm'], scale)}</td>"
            f'<td class="n">{brier}</td></tr>'
        )
    variant_note = f'<p class="meta">Prompt variant: {e(common)}.</p>' if common else ""
    board = (
        "<h2>Per-model coherence leaderboard</h2>"
        f"{variant_note}"
        '<p class="legend"><i style="background:var(--blue)"></i>guaranteed profit per unit of '
        'total stake<i style="background:var(--orange)"></i>after renormalising the four joint '
        "cells</p>"
        '<table><thead><tr><th class="n">Rank</th><th>Model configuration</th>'
        '<th class="n">Tuples</th><th class="n">Profit &ge; 5%</th><th class="n">Mean profit</th>'
        '<th class="n">After renorm.</th><th></th><th class="n">Brier</th></tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
    parts = [
        f'<section id="overview">{cards_html}{caveat}</section>',
        f'<section id="leaderboard">{board}</section>',
    ]
    if figures:
        shots = "".join(embed_png(path, caption, caption) for caption, path in figures)
        parts.append(f'<section id="figures">{shots}</section>')
    if example is not None:
        text = events_list(example_text, example.space.atoms) if example_text else ""
        parts.append(
            f'<section id="example"><h2>A typical tuple</h2><p>{e(example_caption)}</p>'
            f"{text}{book_table(example)}</section>"
        )
    lede = (
        "Language-model forecasters were asked about related events in separate prompts and "
        "scored one question at a time. This report checks whether their answers fit together, "
        "prices the guaranteed arbitrage against them, and measures how much a free, "
        "outcome-independent repair improves their Brier score."
    )
    footer = (
        f"Data: ForecastBench processed forecast sets, snapshot downloaded {e(snapshot.get('downloaded', ''))} "
        f"(SHA-256 <code>{e(str(snapshot.get('sha256', ''))[:16])}</code>), licensed CC BY-SA 4.0; "
        "numbers derived here are shared under the same licence. Karger et al., ForecastBench, ICLR 2025."
    )
    return page(title, "dutchbook · forecastbench", lede, "".join(parts), footer)

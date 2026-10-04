"""Terminal rendering of audits, repairs and extension bounds with ``rich``."""

from __future__ import annotations

from collections.abc import Sequence

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from dutchbook.audit import Audit, Repair
from dutchbook.coherence import Order
from dutchbook.extension import ExtensionBounds

_NORM_TEXT = {
    "l1": "total stake ≤ 1 (profit = sup-norm distance to coherence)",
    "linf": "each stake ≤ 1 (profit = ℓ1 distance to coherence)",
    "l2": "Euclidean stake norm ≤ 1 (profit = ℓ2 distance to coherence)",
}
_UNIT = {
    "l1": "per unit of total stake",
    "linf": "with every stake at most 1",
    "l2": "per unit of Euclidean stake norm",
}
_DISTANCE_NAME = {"linf": "sup-norm", "l1": "ℓ1", "l2": "Euclidean"}


def make_console(width: int | None = None) -> Console:
    return Console(highlight=False, width=width)


def price(p: float) -> str:
    """Probability with at most four decimals, trailing zeros trimmed (keeps two)."""
    s = f"{p:.4f}".rstrip("0")
    return s + "0" * max(0, 2 - (len(s) - s.index(".") - 1))


def signed(x: float, digits: int = 3) -> str:
    return f"{x:+.{digits}f}"


def header_line(console: Console, title: str, parts: Sequence[str]) -> None:
    line = Text()
    line.append(title, style="bold")
    if parts:
        line.append("  ·  " + "  ·  ".join(parts), style="dim")
    console.print(line)


def atom_legend(console: Console, descriptions: dict[str, str], atoms: Sequence[str]) -> None:
    """Print ``atom  description`` lines, truncated to the console width."""
    shown = [(a, descriptions[a]) for a in atoms if a in descriptions]
    if not shown:
        return
    console.print("Events", style="bold")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for atom, text in shown:
        grid.add_row(atom, text)
    console.print(Padding(grid, (0, 0, 0, 2)))


def orders_table(orders: Sequence[Order]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Side")
    table.add_column("Contract on")
    table.add_column("Price", justify="right")
    table.add_column("Stake", justify="right")
    table.add_column("Cash now", justify="right")
    for k, o in enumerate(orders, start=1):
        side = Text(o.side, style="green" if o.side == "BUY" else "red")
        cash = -o.cost
        table.add_row(str(k), side, o.label, price(o.price), f"{o.stake:.3f}", signed(cash))
    return table


def payoff_table(audit: Audit, orders: Sequence[Order], scale: float) -> Table:
    """Net payoff of every bet and in total, in every distinct world."""
    book, space = audit.book, audit.space
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("World")
    per_bet = len(orders) <= 6
    if per_bet:
        for k in range(len(orders)):
            table.add_column(f"#{k + 1}", justify="right", style="dim")
    table.add_column("Net payoff", justify="right")
    q = audit.forecasts
    profit = book.profit * scale
    for w in range(space.W.shape[0]):
        cells = []
        if per_bet:
            for o in orders:
                sign = 1.0 if o.side == "BUY" else -1.0
                cells.append(signed(sign * o.stake * (space.W[w, o.index] - q[o.index])))
        total = book.payoffs[w] * scale
        style = "bold green" if total > 0 else "bold"
        table.add_row(space.describe_world(w), *cells, Text(signed(total), style=style))
    table.add_section()
    label = Text("Guaranteed in every world", style="bold")
    table.add_row(
        label, *([""] * len(orders) if per_bet else []), Text(signed(profit), "bold green")
    )
    return table


def render_audit(
    console: Console,
    audit: Audit,
    stake_norm: str,
    scale: float = 1.0,
    max_worlds: int = 32,
    descriptions: dict[str, str] | None = None,
    group: str | None = None,
) -> None:
    """Verdict, bets at the forecaster's own prices, and the payoff in every world."""
    book, space = audit.book, audit.space
    console.print()
    if group:
        label = f"{group}  ·  {len(audit.labels)} forecasts  ·  {len(space.atoms)} atoms"
        console.print(label, style="dim")
    if descriptions:
        atom_legend(console, descriptions, space.atoms)
    console.print()
    verdict = Text()
    if audit.coherent:
        verdict.append("COHERENT", style="bold green")
        verdict.append("  the forecasts lie in the convex hull of the worlds; no Dutch book exists")
        console.print(verdict)
        if space.W.shape[0] <= max_worlds:
            _witness(console, audit)
        return
    verdict.append("INCOHERENT", style="bold red")
    verdict.append(f"  a Dutch book guarantees {book.profit * scale:.3f}", style="bold")
    verdict.append(
        f" {_UNIT[stake_norm] if scale == 1.0 else f'for stake norm {scale:g}'}", style="dim"
    )
    console.print(verdict)
    console.print(f"  stake norm: {_NORM_TEXT[stake_norm]}", style="dim")
    dists = "  ".join(f"{_DISTANCE_NAME[k]} {v:.3f}" for k, v in audit.distances.items())
    console.print(f"  distance to the nearest coherent forecast: {dists}", style="dim")
    console.print()
    orders = book.orders(audit.forecasts, audit.labels, scale)
    console.print("Bets at the forecaster's own prices", style="bold")
    console.print(orders_table(orders))
    console.print("cash now: paid (-) or received (+) by the bettor at those prices", style="dim")
    console.print()
    if space.W.shape[0] > max_worlds:
        console.print(
            f"Net payoff: at least {book.profit * scale:+.3f} in each of {space.W.shape[0]} worlds "
            f"(listing omitted above {max_worlds} worlds)",
            style="dim",
        )
        return
    heading = (
        "Net payoff in every world" if audit.exhaustive else "Net payoff in the generated worlds"
    )
    console.print(heading, style="bold")
    console.print(payoff_table(audit, orders, scale))
    if not audit.exhaustive:
        console.print(
            "the guarantee is certified over all worlds by the pricing MILP; "
            "only the worlds generated on the way are listed",
            style="dim",
        )


def _witness(console: Console, audit: Audit) -> None:
    """A probability measure over worlds that reproduces the forecasts."""
    book, space = audit.book, audit.space
    console.print()
    console.print(
        "A probability measure over the worlds that realises every forecast", style="bold"
    )
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("World")
    table.add_column("Probability", justify="right")
    for w in range(space.W.shape[0]):
        if book.weights[w] > 1e-9:
            table.add_row(space.describe_world(w), f"{book.weights[w]:.4f}")
    console.print(table)


def render_repair(
    console: Console,
    repair: Repair,
    max_worlds: int = 32,
    descriptions: dict[str, str] | None = None,
    group: str | None = None,
) -> None:
    """Forecasts before and after the projection, and the dominance certificate."""
    cert, space = repair.certificate, repair.space
    console.print()
    kind = "Euclidean (Brier)" if repair.divergence == "brier" else "KL (log score)"
    if group:
        console.print(
            f"{group}  ·  {kind} projection  ·  {len(repair.labels)} forecasts", style="dim"
        )
    if descriptions:
        atom_legend(console, descriptions, space.atoms)
    console.print()
    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    table.add_column("Forecast of")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Change", justify="right")
    for name, before, after in zip(
        repair.labels, repair.projection.q, repair.repaired, strict=True
    ):
        delta = after - before
        style = "dim" if abs(delta) < 5e-7 else ""
        table.add_row(name, price(before), price(after), Text(signed(delta, 4), style=style))
    console.print(table)
    console.print()
    score = "Brier score" if cert.scoring == "brier" else "log loss"
    if cert.divergence <= 1e-12:
        console.print(Text("COHERENT  nothing to repair", style="bold green"))
        return
    console.print(
        Text.assemble(
            ("DOMINATED  ", "bold green" if cert.holds else "bold red"),
            "the repaired vector scores better in every world by at least "
            f"{cert.min_improvement:.4f}",
        )
    )
    console.print(
        f"  guarantee: {score} improves by at least D(p, q) = {cert.divergence:.4f} "
        f"in every world (gap {cert.gap:.0e})",
        style="dim",
    )
    if space.W.shape[0] > max_worlds:
        console.print(
            f"  worst world improves by {cert.min_improvement:.4f}; "
            f"best by {cert.improvement.max():.4f}",
            style="dim",
        )
        return
    console.print()
    wt = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False, header_style="bold")
    wt.add_column("World")
    wt.add_column("Before", justify="right")
    wt.add_column("After", justify="right")
    wt.add_column("Improvement", justify="right")
    wt.add_column("Guaranteed", justify="right")
    for w in range(space.W.shape[0]):
        wt.add_row(
            space.describe_world(w),
            f"{cert.score_before[w]:.4f}",
            f"{cert.score_after[w]:.4f}",
            Text(f"{cert.improvement[w]:.4f}", style="green"),
            f"≥ {cert.divergence:.4f}",
        )
    console.print(wt)


def render_bounds(
    console: Console, target: str, bounds: ExtensionBounds, forecasts: Sequence[str]
) -> None:
    console.print()
    console.print(f"Coherent extension of {len(forecasts)} forecasts to  {target}", style="bold")
    line = Text()
    line.append("  P = ")
    line.append(f"[{bounds.lower:.4f}, {bounds.upper:.4f}]", style="bold")
    line.append(f"   width {bounds.width:.4f}", style="dim")
    console.print(line)
    console.print(
        "  any value outside this interval makes the enlarged system incoherent", style="dim"
    )

"""Pairwise forecast graphs: local versus global coherence.

Questions are the nodes of a graph; every edge carries the four joint-cell forecasts of a
pair of questions, and every node carries one marginal forecast shared by all its edges.
Each pair (two marginals plus four cells) can be checked on its own (*local* coherence),
or the whole graph can be checked against one joint distribution over all questions
(*global* coherence).

* On a **forest**, local coherence implies global coherence: pairwise tables that agree on
  their shared marginals glue along a tree into a single joint distribution (the running
  intersection property).
* On a graph with cycles it does not. Three events that are pairwise "always different"
  are locally coherent for every pair, yet no assignment of three binary values has all
  three pairs different, and a Dutch book exists against the triangle as a whole.

Global coherence factorises over connected components, and the sup-norm distance to the
coherent set of a product is the maximum of the component distances.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dutchbook.audit import Audit, Method, audit
from dutchbook.coherence import StakeNorm
from dutchbook.logic import And, Atom, Formula, Not

CELLS = ("TT", "TF", "FT", "FF")


@dataclass(frozen=True)
class PairwiseSystem:
    """Marginals of questions and joint cells of question pairs.

    Attributes
    ----------
    marginals : dict
        ``P(question)`` for every node.
    cells : dict
        For every edge ``(a, b)``: forecasts of ``(a ∧ b, a ∧ ¬b, ¬a ∧ b, ¬a ∧ ¬b)``.
    """

    marginals: Mapping[str, float]
    cells: Mapping[tuple[str, str], tuple[float, float, float, float]]

    def __post_init__(self) -> None:
        for a, b in self.cells:
            if a not in self.marginals or b not in self.marginals:
                raise ValueError(f"edge {(a, b)} refers to a question without a marginal")
            if a == b:
                raise ValueError("an edge needs two distinct questions")

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.cells)

    def components(self) -> list[PairwiseSystem]:
        """Connected components (isolated nodes are dropped: they constrain nothing)."""
        parent = {n: n for n in self.marginals}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.cells:
            parent[find(a)] = find(b)
        groups: dict[str, list[tuple[str, str]]] = {}
        for edge in self.cells:
            groups.setdefault(find(edge[0]), []).append(edge)
        out = []
        for edges in groups.values():
            nodes = {n for e in edges for n in e}
            out.append(
                PairwiseSystem(
                    {n: self.marginals[n] for n in self.marginals if n in nodes},
                    {e: self.cells[e] for e in edges},
                )
            )
        return out

    def is_forest(self) -> bool:
        """True when the graph has no cycle (``edges = nodes - components``)."""
        nodes = {n for e in self.cells for n in e}
        return len(self.cells) == len(nodes) - len(self.components())

    def formulas(
        self,
    ) -> tuple[list[Formula], list[float], list[str]]:
        """All formulas with their forecasts and display labels (one marginal per node)."""
        nodes = [n for n in self.marginals if any(n in e for e in self.cells)]
        fs: list[Formula] = [Atom(n) for n in nodes]
        q = [float(self.marginals[n]) for n in nodes]
        labels = list(nodes)
        for (a, b), cells in self.cells.items():
            pa, pb = Atom(a), Atom(b)
            for tag, f, v in zip(
                CELLS,
                (And(pa, pb), And(pa, Not(pb)), And(Not(pa), pb), And(Not(pa), Not(pb))),
                cells,
                strict=True,
            ):
                fs.append(f)
                q.append(float(v))
                labels.append(f"{a},{b}:{tag}")
        return fs, q, labels

    def pair_audit(self, edge: tuple[str, str], stake_norm: StakeNorm = "l1") -> Audit:
        """Audit of one edge on its own: its two marginals and four cells."""
        a, b = edge
        sub = PairwiseSystem({a: self.marginals[a], b: self.marginals[b]}, {edge: self.cells[edge]})
        fs, q, labels = sub.formulas()
        return audit(fs, q, labels, stake_norm=stake_norm)


@dataclass(frozen=True)
class ComponentReport:
    """Local and global coherence of one connected component."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    is_forest: bool
    local_profit: tuple[float, ...]
    global_profit: float
    global_audit: Audit

    @property
    def max_local_profit(self) -> float:
        return max(self.local_profit)

    @property
    def locally_coherent(self) -> bool:
        return self.max_local_profit <= 1e-9

    @property
    def globally_coherent(self) -> bool:
        return self.global_profit <= 1e-9


def audit_pairwise(
    system: PairwiseSystem, stake_norm: StakeNorm = "l1", method: Method = "auto"
) -> list[ComponentReport]:
    """Local and global audit of every connected component.

    The global profit is never below the largest local profit: a book against one pair is
    also a book against the whole system.
    """
    reports = []
    for comp in system.components():
        local = tuple(comp.pair_audit(e, stake_norm).book.profit for e in comp.edges)
        fs, q, labels = comp.formulas()
        g = audit(fs, q, labels, stake_norm=stake_norm, method=method)
        nodes = tuple(n for n in comp.marginals)
        reports.append(
            ComponentReport(nodes, comp.edges, comp.is_forest(), local, g.book.profit, g)
        )
    return reports


def fit_cells_to_marginals(
    a: float, b: float, cells: Sequence[float]
) -> tuple[float, float, float, float]:
    """Closest joint table (Euclidean) with exactly the marginals ``P(A) = a``, ``P(B) = b``.

    Joint tables with these marginals form a one-parameter family in ``t = P(A ∧ B)``,
    ``t ∈ [max(0, a + b - 1), min(a, b)]`` (the Fréchet bounds); the squared distance is a
    quadratic in ``t`` minimised in closed form and clipped to that interval.
    """
    c11, c10, c01, c00 = (float(c) for c in cells)
    t = (c11 - c10 - c01 + c00 + 2.0 * a + 2.0 * b - 1.0) / 4.0
    t = min(max(t, max(0.0, a + b - 1.0)), min(a, b))
    table = (t, a - t, b - t, 1.0 - a - b + t)
    return (table[0], max(table[1], 0.0), max(table[2], 0.0), max(table[3], 0.0))


def marginal_consistent(system: PairwiseSystem) -> PairwiseSystem:
    """Replace every edge table by the closest joint table with the nodes' marginals.

    Every pair of the result is locally coherent and the marginals are shared, so any
    remaining incoherence is global: it can only arise on graphs with cycles.
    """
    cells = {
        (a, b): fit_cells_to_marginals(system.marginals[a], system.marginals[b], c)
        for (a, b), c in system.cells.items()
    }
    return PairwiseSystem(system.marginals, cells)


def from_tuples(pairs: Sequence[tuple[str, str]], q: np.ndarray) -> PairwiseSystem:
    """Build a system from combination tuples ``(A, B, A∧B, A∧¬B, ¬A∧B, ¬A∧¬B)``.

    Raises
    ------
    ValueError
        If two tuples disagree about the marginal of a shared question.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape != (len(pairs), 6):
        raise ValueError("q must have one row of six forecasts per pair")
    marginals: dict[str, float] = {}
    cells: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for (a, b), row in zip(pairs, q, strict=True):
        for node, value in ((a, row[0]), (b, row[1])):
            if node in marginals and abs(marginals[node] - value) > 1e-12:
                raise ValueError(f"inconsistent marginal forecasts for question {node!r}")
            marginals[node] = float(value)
        cells[(a, b)] = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
    return PairwiseSystem(marginals, cells)

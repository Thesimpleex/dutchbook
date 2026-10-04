"""Coherence of conditional forecasts via called-off bets.

A forecast ``q_j`` for the conditional event ``E_j | H_j`` prices a bet that pays
``1[E_j] - q_j`` if ``H_j`` is true and is *called off* (pays nothing) otherwise. A bettor
with stakes ``s`` earns ``G_w(s) = sum_j s_j 1[H_j](w) (1[E_j](w) - q_j)`` in world ``w``.

De Finetti's criterion: the assessment is coherent iff no stakes give a gain that is
non-negative in every world and strictly positive in every world where at least one bet
stays on. Checking it is a sequence of linear programs (Coletti and Scozzafava; Gilio;
Biazzo and Gilio). At stage ``alpha`` with active bets ``J``:

* find a probability measure ``lambda`` on the worlds where some ``H_j`` (``j`` in ``J``)
  holds with ``sum_w lambda_w 1[H_j](w) (1[E_j](w) - q_j) = 0`` for ``j`` in ``J``;
* if none exists, the bets in ``J`` form a Dutch book (Gordan's theorem gives the stakes);
* otherwise keep only the bets whose conditioning event has probability zero under *every*
  such measure, and continue; when no bet is left the assessment is coherent.

Bets whose conditioning event can carry positive probability are settled by an earlier
stage, so zero-probability conditioning events are handled without division by zero.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from dutchbook.coherence import DEFAULT_TOL, check_forecasts
from dutchbook.logic import Const, Formula, as_formula, world_space

_HIGHS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}
_GIVEN = re.compile(r"\s+given\s+", re.IGNORECASE)


@dataclass(frozen=True)
class ConditionalEvent:
    """The conditional event ``event | given`` (``given`` is true for an unconditional one)."""

    event: Formula
    given: Formula = Const(True)  # noqa: RUF009 - immutable and shared on purpose

    def __str__(self) -> str:
        if isinstance(self.given, Const) and self.given.value:
            return str(self.event)
        return f"{self.event} | {self.given}"


def parse_conditional(text: str | ConditionalEvent) -> ConditionalEvent:
    """``"B given A"`` to ``B | A``; a string without ``given`` is unconditional."""
    if isinstance(text, ConditionalEvent):
        return text
    parts = _GIVEN.split(text.strip(), maxsplit=1)
    if len(parts) == 2:
        return ConditionalEvent(as_formula(parts[0]), as_formula(parts[1]))
    return ConditionalEvent(as_formula(parts[0]))


@dataclass(frozen=True)
class ConditionalCoherence:
    """Result of :func:`check_conditional_coherence`.

    Attributes
    ----------
    coherent : bool
    stage : int
        Stage at which the check ended (0 is the first linear program).
    active : tuple of int
        Indices of the bets in the failing stage (empty when coherent).
    stages : tuple of tuple
        The active bet indices of every stage that was solved, in order.
    """

    coherent: bool
    stage: int
    active: tuple[int, ...]
    stages: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ConditionalBook:
    """A Dutch book with called-off bets.

    ``payoffs[w]`` is the bettor's gain in world ``w`` (zero where every bet is called off).
    ``profit`` is the minimum gain over the worlds where some bet stays on.
    """

    stakes: np.ndarray
    payoffs: np.ndarray
    profit: float
    active_worlds: np.ndarray


class _Setup:
    """Distinct worlds and the matrices ``H`` (conditioning) and ``A`` (called-off gains)."""

    def __init__(self, events: Sequence[ConditionalEvent | str], q: Sequence[float]) -> None:
        self.events = [parse_conditional(e) for e in events]
        m = len(self.events)
        self.q = check_forecasts(q, m)
        formulas = [e.event for e in self.events] + [e.given for e in self.events]
        self.space = world_space(formulas)
        self.E = self.space.W[:, :m]
        self.H = self.space.W[:, m:]
        self.A = self.H * (self.E - self.q)


def _lp_measure(
    A: np.ndarray, rows: np.ndarray, cols: list[int], objective: np.ndarray | None = None
) -> tuple[bool, np.ndarray | None, float]:
    """Measure on ``rows`` with ``A[rows][:, cols]^T lambda = 0``; optionally maximise a linear
    functional (``objective`` over ``rows``). Returns feasibility, measure, optimum."""
    sub = A[np.ix_(rows, cols)]
    k = rows.size
    a_eq = sparse.csr_matrix(np.vstack([sub.T, np.ones((1, k))]))
    b_eq = np.concatenate([np.zeros(len(cols)), [1.0]])
    c = np.zeros(k) if objective is None else -objective
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * k, method="highs", options=_HIGHS)
    if not res.success:
        return False, None, 0.0
    return True, res.x, float(-res.fun)


def check_conditional_coherence(
    events: Sequence[ConditionalEvent | str],
    forecasts: Sequence[float] | np.ndarray,
    tol: float = DEFAULT_TOL,
) -> ConditionalCoherence:
    """Coherence of conditional forecasts in de Finetti's sense (sequence of LPs)."""
    s = _Setup(events, forecasts)
    m = len(s.events)
    active = list(range(m))
    stages: list[tuple[int, ...]] = []
    stage = 0
    while active:
        stages.append(tuple(active))
        on = np.where((s.H[:, active] > 0.5).any(axis=1))[0]
        ok, _, _ = _lp_measure(s.A, on, active)
        if not ok:
            return ConditionalCoherence(False, stage, tuple(active), tuple(stages))
        settled = []
        for j in active:
            _, _, mass = _lp_measure(s.A, on, active, s.H[on, j])
            if mass > tol:
                settled.append(j)
        active = [j for j in active if j not in settled]
        stage += 1
    return ConditionalCoherence(True, stage, (), tuple(stages))


def find_conditional_dutch_book(
    events: Sequence[ConditionalEvent | str],
    forecasts: Sequence[float] | np.ndarray,
    tol: float = DEFAULT_TOL,
) -> ConditionalBook | None:
    """Stakes (total stake at most 1) maximising the gain guaranteed whenever a bet stays on.

    Returns ``None`` if the assessment is coherent. The gain is non-negative in every world
    and equals at least ``profit > 0`` in every world where one of the failing stage's
    conditioning events holds.
    """
    s = _Setup(events, forecasts)
    verdict = check_conditional_coherence(s.events, s.q, tol)
    if verdict.coherent:
        return None
    cols = list(verdict.active)
    on = np.where((s.H[:, cols] > 0.5).any(axis=1))[0]
    sub = s.A[np.ix_(on, cols)]
    k, c = sub.shape
    # variables [u (c), v (c), t], s = u - v:  max t  s.t.  t <= sub (u - v),  sum(u + v) <= 1
    a_ub = np.vstack(
        [np.hstack([-sub, sub, np.ones((k, 1))]), np.concatenate([np.ones(2 * c), [0.0]])[None, :]]
    )
    b_ub = np.concatenate([np.zeros(k), [1.0]])
    cost = np.zeros(2 * c + 1)
    cost[-1] = -1.0
    res = linprog(
        cost,
        A_ub=sparse.csr_matrix(a_ub),
        b_ub=b_ub,
        bounds=[(0, None)] * (2 * c) + [(None, None)],
        method="highs",
        options=_HIGHS,
    )
    if not res.success:
        raise RuntimeError(f"conditional Dutch-book LP failed: {res.message}")
    stakes = np.zeros(len(s.events))
    stakes[cols] = res.x[:c] - res.x[c : 2 * c]
    payoffs = s.A @ stakes
    return ConditionalBook(stakes, payoffs, float(payoffs[on].min()), on)

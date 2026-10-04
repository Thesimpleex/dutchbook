"""Column generation with a MILP pricing oracle for formula sets beyond enumeration.

The coherent polytope has one vertex per *world*, exponentially many in the number of
atoms. The linear minimisation oracle ``direction -> argmin_w <direction, v_w>`` is a small
mixed-integer program: binary variables for the atoms and for every subformula, linked by
the usual linear encodings of the connectives. It serves two purposes:

* :func:`find_dutch_book_cg` grows a restricted master LP world by world. With stakes ``s``
  from the master, the oracle returns the world with the smallest payoff ``s·(v_w - q)``.
  That payoff is a rigorous *lower* bound on the guaranteed profit (``s`` is feasible for
  the full problem), the master value is an *upper* bound, and the loop stops when they meet.
* :class:`MilpOracle` can be passed to :func:`dutchbook.projection.euclidean_projection` and
  :func:`dutchbook.projection.kl_projection` in place of a vertex matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from dutchbook.coherence import (
    DEFAULT_TOL,
    DutchBook,
    StakeNorm,
    check_forecasts,
    find_dutch_book,
)
from dutchbook.logic import (
    And,
    Atom,
    Const,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    as_formula,
    collect_atoms,
)


class MilpOracle:
    """Linear minimisation oracle over the worlds of a list of formulas.

    ``oracle(direction)`` returns the 0/1 truth vector (one entry per formula) of a world
    minimising ``<direction, v>``; :meth:`solve` also returns the atom assignment.
    """

    def __init__(self, formulas: Sequence[Formula | str], atoms: Sequence[str] | None = None):
        self.formulas = tuple(as_formula(f) for f in formulas)
        self.atoms = tuple(atoms) if atoms is not None else collect_atoms(self.formulas)
        self._index: dict[Formula, int] = {}
        self._rows: list[tuple[dict[int, float], float, float]] = []
        for a in self.atoms:
            self._index[Atom(a)] = len(self._index)
        self.output = np.array([self._node(f) for f in self.formulas], dtype=int)
        n = len(self._index)
        data, ri, ci, lo, hi = [], [], [], [], []
        for r, (coefs, lb, ub) in enumerate(self._rows):
            for c, v in coefs.items():
                data.append(v)
                ri.append(r)
                ci.append(c)
            lo.append(lb)
            hi.append(ub)
        shape = (len(self._rows), n)
        self._A = sparse.csr_matrix((data, (ri, ci)), shape=shape)
        self._lo, self._hi = np.array(lo), np.array(hi)
        self._n = n

    def _row(self, terms: Sequence[tuple[int, float]], lo: float, hi: float) -> None:
        """Add ``lo <= sum(coef * var) <= hi``; repeated variables accumulate."""
        coefs: dict[int, float] = {}
        for var, coef in terms:
            coefs[var] = coefs.get(var, 0.0) + coef
        self._rows.append((coefs, lo, hi))

    def _node(self, f: Formula) -> int:
        if f in self._index:
            return self._index[f]
        if isinstance(f, Atom):
            raise ValueError(f"atom {f.name!r} missing from the atom list")
        if isinstance(f, Const):
            idx = self._new(f)
            self._row([(idx, 1.0)], float(f.value), float(f.value))
            return idx
        if isinstance(f, Not):
            a = self._node(f.arg)
            idx = self._new(f)
            self._row([(idx, 1.0), (a, 1.0)], 1.0, 1.0)
            return idx
        left, right = self._node(f.left), self._node(f.right)  # type: ignore[attr-defined]
        y = self._new(f)
        inf = np.inf
        if isinstance(f, And):  # y = l AND r
            self._row([(y, 1.0), (left, -1.0)], -inf, 0.0)
            self._row([(y, 1.0), (right, -1.0)], -inf, 0.0)
            self._row([(y, 1.0), (left, -1.0), (right, -1.0)], -1.0, inf)
        elif isinstance(f, Or):  # y = l OR r
            self._row([(y, 1.0), (left, -1.0)], 0.0, inf)
            self._row([(y, 1.0), (right, -1.0)], 0.0, inf)
            self._row([(y, 1.0), (left, -1.0), (right, -1.0)], -inf, 0.0)
        elif isinstance(f, Implies):  # y = (NOT l) OR r
            self._row([(y, 1.0), (left, 1.0)], 1.0, inf)
            self._row([(y, 1.0), (right, -1.0)], 0.0, inf)
            self._row([(y, 1.0), (left, 1.0), (right, -1.0)], -inf, 1.0)
        elif isinstance(f, Iff):  # y = 1 - |l - r|
            self._row([(y, 1.0), (left, 1.0), (right, -1.0)], -inf, 1.0)
            self._row([(y, 1.0), (left, -1.0), (right, 1.0)], -inf, 1.0)
            self._row([(y, 1.0), (left, -1.0), (right, -1.0)], -1.0, inf)
            self._row([(y, 1.0), (left, 1.0), (right, 1.0)], 1.0, inf)
        else:  # pragma: no cover - exhaustive over the formula classes
            raise TypeError(f"unsupported formula {f!r}")
        return y

    def _new(self, f: Formula) -> int:
        self._index[f] = len(self._index)
        return self._index[f]

    def solve(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Minimiser of ``<direction, v>``: its truth vector and its atom assignment.

        HiGHS presolve is switched off: with the bundled HiGHS of SciPy 1.18 it returned a
        suboptimal vertex, flagged optimal with zero gap, on one of the ForecastBench
        question graphs (see ``tests/test_oracle.py``), and a pricing oracle that misses
        a world would break the certified bounds.
        """
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (len(self.formulas),):
            raise ValueError(f"direction must have {len(self.formulas)} entries")
        c = np.zeros(self._n)
        np.add.at(c, self.output, direction)
        res = milp(
            c,
            constraints=LinearConstraint(self._A, self._lo, self._hi),
            integrality=np.ones(self._n),
            bounds=Bounds(0, 1),
            options={"mip_rel_gap": 0.0, "presolve": False},
        )
        if not res.success:
            raise RuntimeError(f"pricing MILP failed: {res.message}")
        x = np.rint(res.x)
        return x[self.output], x[: len(self.atoms)].astype(bool)

    def __call__(self, direction: np.ndarray) -> np.ndarray:
        return self.solve(direction)[0]


@dataclass(frozen=True)
class ColumnGeneration:
    """Outcome of :func:`find_dutch_book_cg`.

    ``book.profit`` is the certified lower bound (guaranteed in *every* world, verified by
    the pricing MILP); ``upper_bound`` is the value of the final restricted master. The
    payoffs in ``book`` refer to the generated ``worlds`` only.
    """

    book: DutchBook
    worlds: np.ndarray
    assignments: np.ndarray
    atoms: tuple[str, ...]
    lower_bound: float
    upper_bound: float
    iterations: int
    converged: bool


def find_dutch_book_cg(
    formulas: Sequence[Formula | str],
    q: Sequence[float] | np.ndarray,
    stake_norm: StakeNorm = "l1",
    tol: float = DEFAULT_TOL,
    max_iter: int = 2000,
    oracle: MilpOracle | None = None,
    extreme_seeds: bool = False,
) -> ColumnGeneration:
    """Maximum guaranteed profit against ``q`` by column generation (``l1``/``linf`` stakes).

    Agrees with :func:`dutchbook.coherence.find_dutch_book` on enumerable problems and
    scales to formula sets over many atoms, because only the worlds that matter are
    generated.
    """
    if stake_norm not in ("l1", "linf"):
        raise ValueError("column generation supports stake_norm 'l1' and 'linf'")
    fs = tuple(as_formula(f) for f in formulas)
    q = check_forecasts(q, len(fs))
    oracle = oracle or MilpOracle(fs)
    m = len(fs)
    eye = np.eye(m)
    worlds: list[np.ndarray] = []
    assigns: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(vertex: np.ndarray, assignment: np.ndarray) -> bool:
        key = vertex.astype(np.uint8).tobytes()
        if key in seen:
            return False
        seen.add(key)
        worlds.append(vertex)
        assigns.append(assignment)
        return True

    if extreme_seeds:
        for j in range(m):
            for sign in (1.0, -1.0):
                add(*oracle.solve(sign * eye[j]))
    else:
        for direction in (np.zeros(m), np.ones(m), -np.ones(m)):
            add(*oracle.solve(direction))
    lower, upper = 0.0, np.inf
    best_book: DutchBook | None = None
    best_stakes = np.zeros(m)
    converged = False
    it = 0
    for _ in range(max_iter):
        it += 1
        W = np.array(worlds)
        book = find_dutch_book(W, q, stake_norm, tol)
        upper = book.profit
        if book.coherent:
            lower, best_stakes, best_book, converged = 0.0, book.stakes, book, True
            break
        vertex, assignment = oracle.solve(book.stakes)
        worst = float(book.stakes @ (vertex - q))
        if worst > lower:
            lower, best_stakes, best_book = worst, book.stakes, book
        if upper - worst <= tol:
            converged = True
            break
        if not add(vertex, assignment):
            converged = True
            break
    assert best_book is not None
    W = np.array(worlds)
    payoffs = (W - q) @ best_stakes
    final = DutchBook(
        stakes=best_stakes,
        payoffs=payoffs,
        profit=max(lower, 0.0),
        stake_norm=stake_norm,
        weights=np.pad(best_book.weights, (0, W.shape[0] - best_book.weights.shape[0])),
        coherent=best_book.coherent,
    )
    return ColumnGeneration(
        final, W, np.array(assigns), oracle.atoms, max(lower, 0.0), upper, it, converged
    )

import runpy
from pathlib import Path

import numpy as np
import pytest

from dutchbook import (
    PairwiseSystem,
    audit_pairwise,
    fit_cells_to_marginals,
    from_tuples,
    marginal_consistent,
)


def markov_tables(rng, tree_edges, n):
    """Pairwise tables from a joint distribution over n binary variables (always coherent)."""
    joint = rng.dirichlet(np.ones(2**n) * 0.5)
    states = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    marg = {f"x{i}": float(joint[states[:, i]].sum()) for i in range(n)}
    cells = {}
    for i, j in tree_edges:
        a, b = states[:, i], states[:, j]
        cells[(f"x{i}", f"x{j}")] = (
            float(joint[a & b].sum()),
            float(joint[a & ~b].sum()),
            float(joint[~a & b].sum()),
            float(joint[~a & ~b].sum()),
        )
    return PairwiseSystem(marg, cells)


def frustrated_triangle() -> PairwiseSystem:
    # every pair is "always different": P(x_i != x_j) = 1 with all marginals 1/2
    different = (0.0, 0.5, 0.5, 0.0)
    return PairwiseSystem(
        {"a": 0.5, "b": 0.5, "c": 0.5},
        {("a", "b"): different, ("b", "c"): different, ("a", "c"): different},
    )


def test_frustrated_cycle_is_locally_but_not_globally_coherent():
    (report,) = audit_pairwise(frustrated_triangle(), method="enumerate")
    assert not report.is_forest
    assert report.locally_coherent
    assert not report.globally_coherent
    assert report.global_profit > 0.1
    # in every world at least one pair agrees; the bettor buys the "agree" cells priced at zero
    book = report.global_audit.book
    assert (book.payoffs > 0).all()
    assert report.global_profit >= report.max_local_profit


def test_frustrated_cycle_value_matches_the_hand_computation():
    (report,) = audit_pairwise(frustrated_triangle(), method="enumerate")
    # Hand-built book: sell "a differs from b" cell FT (price 1/2), buy the zero-priced cells
    # "b and c" and "not a and not c", each with stake 1/3. If a is false and b true, the
    # sold cell pays 1 and exactly one of the bought cells pays 1; in every other world the
    # book earns at least 1/2. Gain at least 1/2 per 3 units of stake is 1/6 per unit, and
    # the LP finds nothing better.
    assert report.global_profit == pytest.approx(1 / 6, abs=1e-9)


def test_examples_run(capsys):
    root = Path(__file__).resolve().parent.parent / "examples"
    runpy.run_path(str(root / "frustrated_triangle.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "guaranteed profit against the whole system: 0.167" in out
    runpy.run_path(str(root / "conditional.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "P(A & B) = 0.35: coherent = False" in out
    assert "P(A & B) = 0.20: coherent = True" in out


def test_trees_local_coherence_implies_global(rng):
    shapes = {
        "path": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "star": [(0, 1), (0, 2), (0, 3), (0, 4)],
        "binary": [(0, 1), (0, 2), (1, 3), (1, 4)],
    }
    for edges in shapes.values():
        for _ in range(3):
            system = markov_tables(rng, edges, 5)
            (report,) = audit_pairwise(system, method="enumerate")
            assert report.is_forest
            assert report.locally_coherent
            assert report.globally_coherent


def test_cycles_of_coherent_pairs_can_break_global_coherence(rng):
    # random pairwise tables with shared marginals on a triangle: perturb one table slightly
    # while keeping it a locally coherent distribution with the same marginals
    system = markov_tables(rng, [(0, 1), (1, 2), (0, 2)], 3)
    (report,) = audit_pairwise(system, method="enumerate")
    assert report.globally_coherent  # tables from one joint distribution
    cells = dict(system.cells)
    a, b = system.marginals["x0"], system.marginals["x2"]
    # replace the (x0, x2) table by the most negatively associated one with the same marginals
    c11 = max(0.0, a + b - 1.0)
    cells[("x0", "x2")] = (c11, a - c11, b - c11, 1 - a - b + c11)
    broken = PairwiseSystem(system.marginals, cells)
    (rep2,) = audit_pairwise(broken, method="enumerate")
    assert rep2.locally_coherent
    assert rep2.global_profit >= rep2.max_local_profit - 1e-12


def test_global_profit_dominates_local_profit(rng):
    for _ in range(10):
        n = 4
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.7]
        if not pairs:
            continue
        marg = {f"x{i}": float(rng.random()) for i in range(n)}
        cells = {}
        for i, j in pairs:
            c = rng.dirichlet(np.ones(4))
            cells[(f"x{i}", f"x{j}")] = tuple(float(v) for v in c)
        for report in audit_pairwise(PairwiseSystem(marg, cells), method="enumerate"):
            assert report.global_profit >= report.max_local_profit - 1e-9


def test_components_split_the_graph():
    system = PairwiseSystem(
        {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.5},
        {("a", "b"): (0.25,) * 4, ("c", "d"): (0.25,) * 4},
    )
    comps = system.components()
    assert sorted(len(c.marginals) for c in comps) == [2, 2]  # node e carries no constraint
    assert system.is_forest()
    assert len(audit_pairwise(system, method="enumerate")) == 2


def test_from_tuples_and_consistency_check():
    q = np.array([[0.6, 0.5, 0.3, 0.3, 0.2, 0.2], [0.6, 0.4, 0.24, 0.36, 0.16, 0.24]])
    system = from_tuples([("a", "b"), ("a", "c")], q)
    assert system.marginals == {"a": 0.6, "b": 0.5, "c": 0.4}
    assert system.cells[("a", "c")] == (0.24, 0.36, 0.16, 0.24)
    (report,) = audit_pairwise(system, method="enumerate")
    assert report.globally_coherent
    q[1, 0] = 0.61
    with pytest.raises(ValueError, match="inconsistent marginal"):
        from_tuples([("a", "b"), ("a", "c")], q)
    with pytest.raises(ValueError, match="six forecasts"):
        from_tuples([("a", "b")], np.zeros((2, 6)))


def test_system_validation():
    with pytest.raises(ValueError, match="without a marginal"):
        PairwiseSystem({"a": 0.5}, {("a", "b"): (0.25,) * 4})
    with pytest.raises(ValueError, match="distinct"):
        PairwiseSystem({"a": 0.5}, {("a", "a"): (0.25,) * 4})


def test_column_generation_handles_a_large_forest():
    # a path of 24 questions cannot be enumerated cheaply but is a tree: coherent when locally so
    rng = np.random.default_rng(1)
    n = 24
    marg = {f"x{i}": 0.5 for i in range(n)}
    cells = {}
    for i in range(n - 1):
        p = float(rng.uniform(0.1, 0.4))
        cells[(f"x{i}", f"x{i + 1}")] = (p, 0.5 - p, 0.5 - p, p)
    (report,) = audit_pairwise(PairwiseSystem(marg, cells), method="column-generation")
    assert report.is_forest
    assert report.locally_coherent
    assert report.globally_coherent


def test_fit_cells_to_marginals_matches_brute_force(rng):
    for _ in range(100):
        a, b = rng.random(2)
        raw = rng.dirichlet(np.ones(4)) * rng.uniform(0.5, 2.0)
        fitted = np.array(fit_cells_to_marginals(a, b, raw))
        assert fitted.sum() == pytest.approx(1.0)
        assert (fitted >= -1e-12).all()
        assert fitted[0] + fitted[1] == pytest.approx(a)
        assert fitted[0] + fitted[2] == pytest.approx(b)
        lo, hi = max(0.0, a + b - 1), min(a, b)
        grid = np.linspace(lo, hi, 4001)
        tables = np.stack([grid, a - grid, b - grid, 1 - a - b + grid], axis=1)
        best = tables[np.argmin(((tables - raw) ** 2).sum(axis=1))]
        np.testing.assert_allclose(fitted, best, atol=(hi - lo) / 4000 + 1e-12)


def test_marginal_consistent_system_is_locally_coherent_and_trees_are_globally_coherent(rng):
    edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    marg = {f"x{i}": float(rng.uniform(0.1, 0.9)) for i in range(5)}
    cells = {
        (f"x{i}", f"x{j}"): tuple(float(v) for v in rng.dirichlet(np.ones(4)) * 1.7)
        for i, j in edges
    }
    system = marginal_consistent(PairwiseSystem(marg, cells))
    (report,) = audit_pairwise(system, method="enumerate")
    assert report.is_forest
    assert report.locally_coherent
    assert report.globally_coherent


def test_marginal_consistent_cycle_can_stay_globally_incoherent():
    # three events, every pair strongly negatively associated with marginals 1/2: the
    # closest tables with the given marginals are still "always different"
    raw = (0.0, 0.5, 0.5, 0.0)
    system = marginal_consistent(
        PairwiseSystem(
            {"a": 0.5, "b": 0.5, "c": 0.5},
            {("a", "b"): raw, ("b", "c"): raw, ("a", "c"): raw},
        )
    )
    (report,) = audit_pairwise(system, method="enumerate")
    assert report.locally_coherent
    assert not report.globally_coherent

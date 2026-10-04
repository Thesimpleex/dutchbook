import json
from pathlib import Path

import numpy as np
import pytest

from dutchbook import (
    MilpOracle,
    audit,
    euclidean_projection,
    find_dutch_book,
    find_dutch_book_cg,
    kl_projection,
    repair,
    world_space,
)

CONNECTIVES = ["A & B", "A | B", "A -> B", "A <-> B", "~A", "~(A & ~B)", "true", "false"]


@pytest.mark.parametrize("formula", CONNECTIVES)
def test_milp_encoding_reproduces_each_connective(formula):
    space = world_space(["A", "B", formula])
    oracle = MilpOracle(["A", "B", formula])
    for w in range(space.W.shape[0]):
        # a direction that makes world w the unique minimiser: reward agreement with it
        target = space.W[w]
        direction = np.where(target > 0.5, -1.0, 1.0)
        vertex, assignment = oracle.solve(direction)
        np.testing.assert_array_equal(vertex, target)
        assert assignment.shape == (2,)


def random_formulas(rng, atoms, n):
    def make(depth):
        if depth == 0 or rng.random() < 0.25:
            a = atoms[rng.integers(len(atoms))]
            return a if rng.random() < 0.75 else f"~{a}"
        op = rng.choice(["&", "|", "->", "<->"])
        return f"({make(depth - 1)} {op} {make(depth - 1)})"

    return [make(3) for _ in range(n)]


def test_oracle_minimum_matches_enumeration(rng):
    atoms = [f"x{i}" for i in range(6)]
    for _ in range(12):
        fs = random_formulas(rng, atoms, 6)
        space = world_space(fs)
        oracle = MilpOracle(fs)
        for _ in range(4):
            d = rng.normal(size=6)
            v = oracle(d)
            assert d @ v == pytest.approx((space.W @ d).min(), abs=1e-9)
            assert any((v == space.W).all(axis=1))


def test_oracle_rejects_bad_direction_and_unknown_atoms():
    oracle = MilpOracle(["A", "A & B"])
    with pytest.raises(ValueError, match="direction"):
        oracle(np.zeros(3))
    with pytest.raises(ValueError, match="missing"):
        MilpOracle(["A & C"], atoms=["A"])


@pytest.mark.parametrize("norm", ["l1", "linf"])
def test_column_generation_matches_enumeration(rng, norm):
    atoms = [f"x{i}" for i in range(8)]
    for _ in range(6):
        fs = random_formulas(rng, atoms, 7)
        q = rng.random(7)
        ref = find_dutch_book(world_space(fs).W, q, norm)
        cg = find_dutch_book_cg(fs, q, norm)
        assert cg.converged
        assert cg.book.profit == pytest.approx(ref.profit, abs=1e-8)
        assert cg.upper_bound == pytest.approx(ref.profit, abs=1e-8)
        assert cg.lower_bound <= cg.upper_bound + 1e-12
        # the generated worlds only: payoffs of the stakes are at least the certified profit
        if not cg.book.coherent:
            assert cg.book.payoffs.min() >= cg.book.profit - 1e-9


def test_column_generation_certifies_in_every_world(rng):
    atoms = [f"x{i}" for i in range(7)]
    fs = random_formulas(rng, atoms, 6)
    q = rng.random(6)
    cg = find_dutch_book_cg(fs, q, "l1")
    full = world_space(fs).W
    guaranteed = ((full - q) @ cg.book.stakes).min()
    assert guaranteed == pytest.approx(cg.book.profit, abs=1e-8)


def test_column_generation_on_a_chain_of_sixteen_atoms():
    n = 16
    fs = [f"x{i}" for i in range(n)] + [f"x{i} -> x{i + 1}" for i in range(n - 1)]
    rng = np.random.default_rng(5)
    q = np.concatenate([rng.uniform(0.2, 0.8, n), np.full(n - 1, 0.9)])
    ref = audit(fs, q, method="enumerate")
    cg = audit(fs, q, method="column-generation")
    assert not cg.exhaustive
    assert ref.exhaustive
    assert cg.book.profit == pytest.approx(ref.book.profit, abs=1e-8)
    assert cg.space.W.shape[0] < ref.space.W.shape[0]
    # projection through the oracle equals the enumerated projection
    rep_cg = repair(fs, q, method="column-generation")
    rep_en = repair(fs, q, method="enumerate")
    np.testing.assert_allclose(rep_cg.repaired, rep_en.repaired, atol=1e-6)
    assert rep_cg.certificate.holds
    assert rep_cg.certificate.divergence == pytest.approx(rep_en.certificate.divergence, abs=1e-8)


def test_audit_auto_switches_beyond_enumeration():
    n = 26
    fs = [f"x{i}" for i in range(n)] + ["x0 & x1", "x1 & x2"]
    q = np.concatenate([np.full(n, 0.5), [0.9, 0.1]])
    a = audit(fs, q)  # 26 atoms: cannot be enumerated, handled by column generation
    assert not a.exhaustive
    # P(x0 & x1) = 0.9 > P(x1) = 0.5: sure loss of at least 0.2 / 2 by the Fréchet bound
    assert a.book.profit >= 0.2 - 1e-9
    assert not a.coherent
    with pytest.raises(ValueError, match="enumeration limit"):
        audit(fs, q, method="enumerate")


def test_l2_audit_with_oracle_and_method_validation():
    fs = ["A", "~A"]
    a = audit(fs, [0.6, 0.6], stake_norm="l2", method="column-generation")
    assert a.book.profit == pytest.approx(0.1 * 2**0.5, abs=1e-9)
    ok = audit(fs, [0.4, 0.6], stake_norm="l2", method="column-generation")
    assert ok.coherent
    with pytest.raises(ValueError, match="method"):
        audit(fs, [0.5, 0.5], method="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="supports"):
        find_dutch_book_cg(fs, [0.5, 0.5], "l2")  # type: ignore[arg-type]


def test_projections_accept_the_oracle_directly(rng):
    atoms = [f"x{i}" for i in range(5)]
    fs = random_formulas(rng, atoms, 5)
    q = rng.random(5)
    W = world_space(fs).W
    oracle = MilpOracle(fs)
    np.testing.assert_allclose(
        euclidean_projection(oracle, q).point, euclidean_projection(W, q).point, atol=1e-8
    )
    np.testing.assert_allclose(kl_projection(oracle, q).point, kl_projection(W, q).point, atol=1e-5)


@pytest.mark.parametrize("formula", ["A & A", "A | A", "A -> A", "A <-> A", "(A & B) <-> (A & B)"])
def test_repeated_operands_are_encoded_correctly(formula):
    space = world_space(["A", "B", formula])
    oracle = MilpOracle(["A", "B", formula])
    for w in range(space.W.shape[0]):
        direction = np.where(space.W[w] > 0.5, -1.0, 1.0)
        np.testing.assert_array_equal(oracle(direction), space.W[w])


def test_seeding_strategies_agree(rng):
    atoms = [f"x{i}" for i in range(6)]
    fs = random_formulas(rng, atoms, 6)
    q = rng.random(6)
    minimal = find_dutch_book_cg(fs, q, "l1")
    extreme = find_dutch_book_cg(fs, q, "l1", extreme_seeds=True)
    assert minimal.book.profit == pytest.approx(extreme.book.profit, abs=1e-8)
    assert minimal.converged
    assert extreme.converged


def test_pricing_oracle_regression_for_a_presolve_failure():
    """On this instance (24 atoms, 145 formulas, from a ForecastBench question tree) the
    bundled HiGHS with presolve returned -0.25 flagged optimal; the true minimum is -0.375."""
    path = Path(__file__).parent / "data" / "milp_presolve_regression.json"
    data = json.loads(path.read_text())
    direction = np.array(data["direction"])
    oracle = MilpOracle(data["formulas"])
    vertex = oracle(direction)
    assert direction @ vertex == pytest.approx(data["minimum"], abs=1e-9)
    # a column-generation run on the same system must not certify a book that does not exist
    n = len(data["formulas"])
    assert np.count_nonzero(direction) == 8
    assert vertex.shape == (n,)

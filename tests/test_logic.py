import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from dutchbook.logic import (
    And,
    Atom,
    Const,
    Formula,
    FormulaSyntaxError,
    Iff,
    Implies,
    Not,
    Or,
    enumerate_worlds,
    parse,
    world_space,
)

A, B, C = Atom("A"), Atom("B"), Atom("C")


def test_atoms_and_precedence():
    assert parse("A") == A
    assert parse("~A & B | C") == Or(And(Not(A), B), C)
    assert parse("A | B & C") == Or(A, And(B, C))
    assert parse("A -> B -> C") == Implies(A, Implies(B, C))
    assert parse("A & B -> C | A") == Implies(And(A, B), Or(C, A))
    assert parse("A <-> B -> C") == Iff(A, Implies(B, C))
    assert parse("A <-> B <-> C") == Iff(Iff(A, B), C)
    assert parse("not not A") == Not(Not(A))


@pytest.mark.parametrize(
    "text",
    ["A and B", "A && B", "A ∧ B", "A & B"],
)
def test_conjunction_spellings(text):
    assert parse(text) == And(A, B)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A or B", Or(A, B)),
        ("A ∨ B", Or(A, B)),
        ("A => B", Implies(A, B)),
        ("A implies B", Implies(A, B)),
        ("A → B", Implies(A, B)),
        ("A iff B", Iff(A, B)),
        ("A ↔ B", Iff(A, B)),
        ("A <=> B", Iff(A, B)),
        ("!A", Not(A)),
        ("¬A", Not(A)),
        ("true & A", And(Const(True), A)),
        ("⊥ | A", Or(Const(False), A)),
    ],
)
def test_other_spellings(text, expected):
    assert parse(text) == expected


def test_unicode_pretty_printing():
    assert str(parse("~A & (B | C)")) == "¬A ∧ (B ∨ C)"
    assert str(parse("(A -> B) -> C")) == "(A → B) → C"
    assert str(parse("A -> (B -> C)")) == "A → B → C"
    assert str(parse("A & (B & C)")) == "A ∧ (B ∧ C)"
    assert str(parse("(A & B) & C")) == "A ∧ B ∧ C"
    assert str(parse("~(A & B)")) == "¬(A ∧ B)"
    assert parse("A <-> ~B").ascii() == "A <-> ~B"


@pytest.mark.parametrize(
    ("text", "caret"),
    [("A &", 3), ("(A | B", 6), ("A B", 2), ("A $ B", 2), ("", None), ("& A", 0), ("A )", 2)],
)
def test_syntax_errors_point_at_the_problem(text, caret):
    with pytest.raises(FormulaSyntaxError) as exc:
        parse(text)
    if caret is not None:
        assert f"position {caret}" in str(exc.value)


def test_evaluate_truth_tables():
    worlds = enumerate_worlds(2)
    values = {"A": worlds[:, 0], "B": worlds[:, 1]}
    # world order: TT, TF, FT, FF
    assert parse("A & B").evaluate(values).tolist() == [True, False, False, False]
    assert parse("A | B").evaluate(values).tolist() == [True, True, True, False]
    assert parse("A -> B").evaluate(values).tolist() == [True, False, True, True]
    assert parse("A <-> B").evaluate(values).tolist() == [True, False, False, True]
    assert parse("~A").evaluate(values).tolist() == [False, False, True, True]


def test_evaluate_scalar_and_missing_atom():
    assert bool(parse("A & ~B").evaluate({"A": True, "B": False}))
    with pytest.raises(KeyError, match="no truth value"):
        parse("A & C").evaluate({"A": True})


def test_enumerate_worlds_order_and_limits():
    w = enumerate_worlds(3)
    assert w.shape == (8, 3)
    assert w[0].all()
    assert not w[-1].any()
    assert len({tuple(r) for r in w}) == 8
    assert enumerate_worlds(0).shape == (1, 0)
    with pytest.raises(ValueError, match="enumeration limit"):
        enumerate_worlds(40)


def test_enumerate_worlds_twenty_atoms():
    w = enumerate_worlds(20)
    assert w.shape == (2**20, 20)
    assert w[:, 0].sum() == 2**19
    assert w[:, 19].sum() == 2**19


def test_world_space_merges_equivalent_worlds():
    space = world_space(["A", "A & B"])
    # atoms A, B: patterns (1,1), (1,0), (0,0) -> the two worlds with A false coincide
    assert space.W.shape == (3, 2)
    assert space.atoms == ("A", "B")
    assert space.describe_world(0) == "A ∧ B"
    assert space.describe_world(1) == "A ∧ ¬B"
    assert space.describe_world(2) == "¬A ∧ B"


def test_world_space_atom_order_and_errors():
    space = world_space(["B", "A"], atoms=["A", "B"])
    assert space.atoms == ("A", "B")
    with pytest.raises(ValueError, match="missing"):
        world_space(["A & C"], atoms=["A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        world_space(["A"], atoms=["A", "A"])


def test_world_space_many_formulas_uses_byte_keys():
    formulas = [f"x{i % 3} & x{(i + 1) % 4}" for i in range(70)]
    space = world_space(formulas)
    assert space.W.shape[1] == 70
    assert np.unique(space.W, axis=0).shape[0] == space.W.shape[0]


def test_world_space_tautology_and_contradiction_columns():
    space = world_space(["A | ~A", "A & ~A", "A"])
    assert (space.W[:, 0] == 1).all()
    assert (space.W[:, 1] == 0).all()
    assert space.W.shape[0] == 2


_ATOM_NAMES = ["p", "q", "r"]


def _formulas() -> st.SearchStrategy[Formula]:
    leaf = st.sampled_from(_ATOM_NAMES).map(Atom) | st.booleans().map(Const)
    return st.recursive(
        leaf,
        lambda sub: (
            sub.map(Not)
            | st.tuples(sub, sub).map(lambda t: And(*t))
            | st.tuples(sub, sub).map(lambda t: Or(*t))
            | st.tuples(sub, sub).map(lambda t: Implies(*t))
            | st.tuples(sub, sub).map(lambda t: Iff(*t))
        ),
        max_leaves=8,
    )


@given(_formulas())
def test_print_parse_round_trip(f):
    assert parse(str(f)) == f
    assert parse(f.ascii()) == f


@given(_formulas(), _formulas())
def test_semantics_of_connectives(f, g):
    worlds = enumerate_worlds(3)
    vals = {a: worlds[:, i] for i, a in enumerate(_ATOM_NAMES)}
    ev = f.evaluate(vals), g.evaluate(vals)
    n = worlds.shape[0]
    ev = tuple(np.broadcast_to(e, (n,)) for e in ev)
    assert ((f & g).evaluate(vals) == (ev[0] & ev[1])).all()
    assert ((f | g).evaluate(vals) == (ev[0] | ev[1])).all()
    assert ((f >> g).evaluate(vals) == (~ev[0] | ev[1])).all()
    assert (f.iff(g).evaluate(vals) == (ev[0] == ev[1])).all()


def test_de_morgan_over_all_worlds():
    worlds = enumerate_worlds(2)
    vals = {"A": worlds[:, 0], "B": worlds[:, 1]}
    for a, b in itertools.product([parse("A"), parse("~A")], [parse("B"), parse("A | B")]):
        lhs = (~(a & b)).evaluate(vals)
        rhs = ((~a) | (~b)).evaluate(vals)
        assert (lhs == rhs).all()

import numpy as np
import pytest
from scipy.optimize import linprog

from dutchbook import (
    ConditionalEvent,
    check_conditional_coherence,
    find_conditional_dutch_book,
    parse,
    parse_conditional,
)


def test_parse_conditional():
    c = parse_conditional("B & C given A | D")
    assert c.event == parse("B & C")
    assert c.given == parse("A | D")
    assert str(c) == "B ∧ C | A ∨ D"
    assert str(parse_conditional("A")) == "A"
    assert parse_conditional(c) is c
    assert parse_conditional("B GIVEN A").given == parse("A")


def test_chain_rule_textbook_example():
    # P(A) = y, P(B | A) = x force P(A & B) = x y; any other price is a Dutch book
    x, y = 0.4, 0.5
    events = ["A", "B given A", "A & B"]
    assert check_conditional_coherence(events, [y, x, x * y]).coherent
    for z in (0.1, 0.3, 0.35):
        verdict = check_conditional_coherence(events, [y, x, z])
        assert not verdict.coherent
        book = find_conditional_dutch_book(events, [y, x, z])
        assert book is not None
        assert (book.payoffs >= -1e-9).all()
        assert (book.payoffs[book.active_worlds] > 1e-6).all()
        assert book.profit > 0
        assert abs(book.stakes).sum() == pytest.approx(1.0, abs=1e-9)


def test_chain_rule_book_needs_the_called_off_bet():
    events = ["A", "B given A", "A & B"]
    book = find_conditional_dutch_book(events, [0.5, 0.4, 0.35])
    assert book is not None
    assert abs(book.stakes[1]) > 1e-6  # the conditional bet is part of the book
    # the unconditional forecasts alone are coherent, so they cannot form a book without it
    assert check_conditional_coherence([events[0], events[2]], [0.5, 0.35]).coherent
    # and where A is false that bet pays nothing: only the unconditional bets are paid
    q = np.array([0.5, 0.4, 0.35])
    gain_not_a = book.stakes[0] * (0 - q[0]) + book.stakes[2] * (0 - q[2])
    assert gain_not_a in [pytest.approx(g) for g in book.payoffs]


def test_bayes_theorem_consistency():
    # P(B|A) P(A) = P(A|B) P(B) is necessary for coherence of the four assessments
    events = ["B given A", "A given B", "A", "B"]
    pa, pb, pab = 0.4, 0.5, 0.15
    good = [pab / pa, pab / pb, pa, pb]
    assert check_conditional_coherence(events, good).coherent
    bad = [0.6, pab / pb, pa, pb]
    assert not check_conditional_coherence(events, bad).coherent


def test_two_prices_for_the_same_conditional_event_are_incoherent():
    verdict = check_conditional_coherence(["B given A", "B given A"], [0.3, 0.7])
    assert not verdict.coherent
    assert verdict.stage == 0
    assert check_conditional_coherence(["B given A", "B given A"], [0.3, 0.3]).coherent


def test_zero_probability_conditioning_needs_the_second_stage():
    # P(A) = 0 with two different prices for B given A: the first LP is feasible (put all
    # mass on not-A) but the second stage, restricted to A, is not
    events = ["A", "B given A", "B given A"]
    verdict = check_conditional_coherence(events, [0.0, 0.3, 0.7])
    assert not verdict.coherent
    assert verdict.stage == 1
    assert verdict.active == (1, 2)
    assert verdict.stages == ((0, 1, 2), (1, 2))
    book = find_conditional_dutch_book(events, [0.0, 0.3, 0.7])
    assert book is not None
    assert book.profit > 0
    assert (book.payoffs >= -1e-9).all()
    # a null conditioning event with a single conditional price is fine
    assert check_conditional_coherence(["A", "B given A"], [0.0, 0.3]).coherent


def test_certain_conditioning_event_reduces_to_unconditional_coherence():
    events = ["B given A | ~A", "~B given A | ~A"]
    assert check_conditional_coherence(events, [0.4, 0.6]).coherent
    assert not check_conditional_coherence(events, [0.4, 0.7]).coherent


def test_stage_zero_agrees_with_the_theorem_of_the_alternative(rng):
    """Stage 0 fails iff stakes exist with strictly positive gain in every world where some
    bet is on (checked by a separate LP on the dual side)."""
    events = ["B given A", "C given B", "A & C", "A", "C given A | B"]
    for _ in range(40):
        q = rng.random(5)
        verdict = check_conditional_coherence(events, q)
        book = find_conditional_dutch_book(events, q)
        assert (book is None) == verdict.coherent
        if verdict.stage == 0 and not verdict.coherent:
            from dutchbook.conditional import _Setup

            s = _Setup(events, q)
            on = np.where((s.H > 0.5).any(axis=1))[0]
            res = linprog(
                np.zeros(5),
                A_ub=-s.A[on],
                b_ub=-np.ones(on.size),
                bounds=[(None, None)] * 5,
                method="highs",
            )
            assert res.success  # an s with gain >= 1 in every active world exists


def test_assessments_from_a_positive_measure_are_coherent(rng):
    from dutchbook import world_space

    events = ["B given A", "A given B | C", "C", "B & C given A"]
    parsed = [parse_conditional(e) for e in events]
    space = world_space([p.event for p in parsed] + [p.given for p in parsed] + ["A", "B", "C"])
    atoms = space.atoms
    worlds = space.assignments
    for _ in range(25):
        mu = rng.dirichlet(np.ones(worlds.shape[0]))
        values = {a: worlds[:, i] for i, a in enumerate(atoms)}
        q = []
        for p in parsed:
            e = np.broadcast_to(p.event.evaluate(values), mu.shape)
            h = np.broadcast_to(p.given.evaluate(values), mu.shape)
            q.append(float(mu[e & h].sum() / mu[h].sum()))
        assert check_conditional_coherence(events, q).coherent
        assert find_conditional_dutch_book(events, q) is None


def test_conditional_event_object_and_validation():
    ev = ConditionalEvent(parse("B"), parse("A"))
    assert check_conditional_coherence([ev, "A"], [0.2, 0.5]).coherent
    with pytest.raises(ValueError, match="expected 2 forecasts"):
        check_conditional_coherence([ev, "A"], [0.2])
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        check_conditional_coherence([ev], [1.5])

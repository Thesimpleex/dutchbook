import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dutchbook import IncoherentForecastsError, extension_bounds, is_coherent, parse, world_space


def test_modus_ponens_bounds():
    # P(A) = a and P(A & B) = a * P(B|A): P(B) lies in [P(A&B), P(A&B) + 1 - P(A)]
    for a, c in [(0.7, 0.5), (0.3, 0.9), (1.0, 0.4), (0.5, 0.0)]:
        bounds = extension_bounds(["A", "A & B"], [a, a * c], "B")
        assert bounds.lower == pytest.approx(a * c, abs=1e-9)
        assert bounds.upper == pytest.approx(a * c + 1 - a, abs=1e-9)


def test_modus_ponens_with_an_implication_premise():
    # premises A and A -> B with probabilities x and y; P(B) >= x + y - 1
    bounds = extension_bounds(["A", "A -> B"], [0.8, 0.9], "B")
    assert bounds.lower == pytest.approx(0.7, abs=1e-9)
    assert bounds.upper == pytest.approx(0.9, abs=1e-9)


def test_frechet_bounds_for_a_conjunction():
    for a, b in [(0.6, 0.7), (0.2, 0.3), (0.9, 0.9)]:
        bounds = extension_bounds(["A", "B"], [a, b], "A & B")
        assert bounds.lower == pytest.approx(max(0.0, a + b - 1), abs=1e-9)
        assert bounds.upper == pytest.approx(min(a, b), abs=1e-9)
        union = extension_bounds(["A", "B"], [a, b], "A | B")
        assert union.lower == pytest.approx(max(a, b), abs=1e-9)
        assert union.upper == pytest.approx(min(1.0, a + b), abs=1e-9)


def test_bounds_are_attained_by_world_measures():
    bounds = extension_bounds(["A", "B"], [0.6, 0.7], "A & B")
    space = bounds.space
    t = space.W[:, 2]
    assert bounds.lower_measure @ t == pytest.approx(bounds.lower, abs=1e-9)
    assert bounds.upper_measure @ t == pytest.approx(bounds.upper, abs=1e-9)
    for measure in (bounds.lower_measure, bounds.upper_measure):
        np.testing.assert_allclose(space.W[:, :2].T @ measure, [0.6, 0.7], atol=1e-9)
        assert measure.sum() == pytest.approx(1.0, abs=1e-9)


def test_bounds_are_tight_in_the_coherence_sense():
    forecasts = [0.8, 0.5]
    bounds = extension_bounds(["A", "A -> B"], [0.8, 0.5], "B")
    space = world_space(["A", "A -> B", "B"])
    for value, expected in [
        (bounds.lower, True),
        (bounds.upper, True),
        (bounds.lower - 0.01, False),
        (bounds.upper + 0.01, False),
        ((bounds.lower + bounds.upper) / 2, True),
    ]:
        assert is_coherent(space.W, [*forecasts, value], tol=1e-8) is expected


def test_formula_already_forecast_has_a_degenerate_interval():
    bounds = extension_bounds(["A", "A | B"], [0.4, 0.7], "A | B")
    assert bounds.lower == pytest.approx(0.7, abs=1e-9)
    assert bounds.upper == pytest.approx(0.7, abs=1e-9)
    assert bounds.width == pytest.approx(0.0, abs=1e-9)


def test_tautology_and_contradiction():
    assert extension_bounds(["A"], [0.3], "A | ~A").lower == pytest.approx(1.0)
    assert extension_bounds(["A"], [0.3], "A & ~A").upper == pytest.approx(0.0)


def test_incoherent_forecasts_are_refused():
    with pytest.raises(IncoherentForecastsError) as exc:
        extension_bounds(["A", "~A"], [0.6, 0.6], "A")
    assert exc.value.distance == pytest.approx(0.1)
    assert "incoherent" in str(exc.value)


def test_accepts_formula_objects_and_floating_point_products():
    a, c = 0.7, 0.35
    bounds = extension_bounds([parse("A"), parse("A & B")], [a, a * c], parse("B"))
    assert bounds.lower == pytest.approx(a * c, abs=1e-9)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_property_extension_bounds_contain_the_generating_value(seed):
    rng = np.random.default_rng(seed)
    formulas = ["A", "B | C", "A & ~B", "C -> A"]
    target = "B & C"
    space = world_space([*formulas, target])
    mu = rng.dirichlet(np.ones(space.W.shape[0]) * 0.7)
    q = np.clip(space.W[:, :4].T @ mu, 0, 1)
    truth = float(space.W[:, 4] @ mu)
    bounds = extension_bounds(formulas, q, target)
    assert bounds.lower - 1e-8 <= truth <= bounds.upper + 1e-8

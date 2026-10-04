import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from conftest import random_truth_matrix
from dutchbook import audit, coherence_distance, find_dutch_book, is_coherent, world_space
from dutchbook.coherence import check_forecasts, check_matrix, stake_norm_value

AND_NOT = world_space(["A", "~A"]).W


def test_a_and_not_a_closed_form():
    # prices 0.6 and 0.6 sum to 1.2: sell both contracts, collect 1.2, pay 1 in every world
    for total in (0.9, 1.2, 1.5):
        q = np.array([total / 2, total / 2])
        excess = abs(total - 1.0)
        assert coherence_distance(AND_NOT, q, "linf").value == pytest.approx(excess / 2, abs=1e-12)
        assert coherence_distance(AND_NOT, q, "l1").value == pytest.approx(excess, abs=1e-12)
        assert coherence_distance(AND_NOT, q, "l2").value == pytest.approx(
            excess / 2**0.5, abs=1e-12
        )
    book = find_dutch_book(AND_NOT, [0.6, 0.6], "l1")
    np.testing.assert_allclose(book.stakes, [-0.5, -0.5], atol=1e-12)
    assert book.profit == pytest.approx(0.1, abs=1e-12)
    np.testing.assert_allclose(book.payoffs, [0.1, 0.1], atol=1e-12)


def test_underpriced_pair_is_bought():
    book = find_dutch_book(AND_NOT, [0.3, 0.3], "l1")
    assert (book.stakes > 0).all()
    assert (book.payoffs > 0).all()


def test_coherent_forecasts_have_no_book():
    assert is_coherent(AND_NOT, [0.37, 0.63])
    book = find_dutch_book(AND_NOT, [0.37, 0.63])
    assert book.coherent
    assert not book.stakes.any()
    assert book.profit == 0.0


def test_linda_conjunction_fallacy():
    space = world_space(["B", "F & B"])
    q = np.array([0.15, 0.35])  # a conjunction judged more likely than its conjunct
    for norm, expected in (("l1", 0.10), ("linf", 0.20), ("l2", 0.10 * 2**0.5)):
        book = find_dutch_book(space.W, q, norm)
        assert book.profit == pytest.approx(expected, abs=1e-12)
        assert (book.payoffs >= book.profit - 1e-12).all()
    book = find_dutch_book(space.W, q, "l1")
    sides = {o.label: o.side for o in book.orders(q, ["B", "F & B"])}
    assert sides == {"B": "BUY", "F & B": "SELL"}


def test_payoffs_are_positive_in_every_world(rng):
    for _ in range(50):
        W = random_truth_matrix(rng, 3, 5)
        q = rng.random(5)
        for norm in ("l1", "linf", "l2"):
            book = find_dutch_book(W, q, norm)
            if not book.coherent:
                assert (book.payoffs > 0).all()
                assert book.payoffs.min() == pytest.approx(book.profit, abs=1e-12)
                # the payoffs are exactly the net payoffs of the stakes
                np.testing.assert_allclose(book.payoffs, (W - q) @ book.stakes, atol=1e-12)


@pytest.mark.parametrize(
    ("stake_norm", "dist_norm"), [("l1", "linf"), ("linf", "l1"), ("l2", "l2")]
)
def test_strong_duality_on_random_instances(rng, stake_norm, dist_norm):
    for _ in range(60):
        W = random_truth_matrix(rng, int(rng.integers(2, 5)), int(rng.integers(2, 7)))
        q = rng.random(W.shape[1])
        book = find_dutch_book(W, q, stake_norm)
        dist = coherence_distance(W, q, dist_norm)
        assert book.profit == pytest.approx(dist.value, abs=1e-9)
        assert stake_norm_value(book.stakes, stake_norm) <= 1.0 + 1e-9
        if not book.coherent:
            assert stake_norm_value(book.stakes, stake_norm) == pytest.approx(1.0, abs=1e-9)
        # the dual weights are a probability measure over worlds realising the distance
        assert book.weights.sum() == pytest.approx(1.0)
        assert (book.weights >= 0).all()
        gap = W.T @ book.weights - q
        realised = {"linf": np.abs(gap).max(), "l1": np.abs(gap).sum(), "l2": np.linalg.norm(gap)}
        assert realised[dist_norm] == pytest.approx(dist.value, abs=1e-8)


def slsqp_linf_distance(W: np.ndarray, q: np.ndarray) -> float:
    """Epigraph form of the sup-norm distance, solved by sequential quadratic programming."""
    k = W.shape[0]
    x0 = np.append(np.full(k, 1 / k), 1.0)

    def residual(x: np.ndarray) -> np.ndarray:
        return W.T @ x[:k] - q

    cons = [
        {"type": "eq", "fun": lambda x: x[:k].sum() - 1},
        {"type": "ineq", "fun": lambda x: x[k] - residual(x)},
        {"type": "ineq", "fun": lambda x: x[k] + residual(x)},
    ]
    res = minimize(
        lambda x: x[k],
        x0,
        jac=lambda _: np.append(np.zeros(k), 1.0),
        bounds=[(0, 1)] * k + [(0, 5)],
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    return float(res.fun)


def test_linf_distance_against_independent_nonlinear_solver(rng):
    for _ in range(25):
        W = random_truth_matrix(rng, 3, 4)
        q = rng.random(4)
        assert coherence_distance(W, q, "linf").value == pytest.approx(
            slsqp_linf_distance(W, q), abs=1e-7
        )


def test_distance_zero_iff_coherent(rng):
    W = random_truth_matrix(rng, 3, 4)
    lam = rng.dirichlet(np.ones(W.shape[0]))
    q = W.T @ lam
    for norm in ("linf", "l1", "l2"):
        assert coherence_distance(W, q, norm).value < 1e-9
    assert is_coherent(W, q)
    q_bad = q.copy()
    q_bad[0] = min(1.0, q_bad[0] + 0.2) if q_bad[0] < 0.8 else q_bad[0] - 0.2
    # moving one coordinate generally leaves the polytope; check LP agrees with the dual book
    assert is_coherent(W, q_bad) == find_dutch_book(W, q_bad).coherent


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_property_coherent_inputs_have_zero_sure_loss(seed):
    rng = np.random.default_rng(seed)
    W = random_truth_matrix(rng, int(rng.integers(2, 5)), int(rng.integers(2, 8)))
    q = W.T @ rng.dirichlet(np.ones(W.shape[0]) * rng.uniform(0.2, 2))
    q = np.clip(q, 0.0, 1.0)
    for norm in ("l1", "linf", "l2"):
        book = find_dutch_book(W, q, norm)
        assert book.coherent
        assert book.profit == 0.0


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_property_permutation_and_complement_invariance(seed):
    rng = np.random.default_rng(seed)
    W = random_truth_matrix(rng, 3, 5)
    q = rng.random(5)
    base = {n: coherence_distance(W, q, n).value for n in ("linf", "l1", "l2")}
    # permuting formulas and worlds changes nothing
    pf, pw = rng.permutation(5), rng.permutation(W.shape[0])
    for n in base:
        assert coherence_distance(W[pw][:, pf], q[pf], n).value == pytest.approx(base[n], abs=1e-9)
    # replacing every formula by its negation maps q to 1 - q and W to 1 - W
    for n in base:
        assert coherence_distance(1 - W, 1 - q, n).value == pytest.approx(base[n], abs=1e-9)
    # scaling the stake budget scales the profit linearly
    book = find_dutch_book(W, q, "l1")
    scaled = book.orders(q, scale=7.0)
    assert sum(o.stake for o in scaled) == pytest.approx(7.0 * np.abs(book.stakes).sum(), abs=1e-8)


def test_orders_cash_flows():
    book = find_dutch_book(AND_NOT, [0.6, 0.6], "l1")
    orders = book.orders([0.6, 0.6], ["A", "~A"], scale=100.0)
    assert [o.side for o in orders] == ["SELL", "SELL"]
    assert sum(-o.cost for o in orders) == pytest.approx(60.0)  # cash received


def test_input_validation():
    with pytest.raises(ValueError, match="0 and 1"):
        check_matrix(np.array([[0.5, 1.0]]))
    with pytest.raises(ValueError, match="non-empty"):
        check_matrix(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        check_forecasts([0.5, 1.2], 2)
    with pytest.raises(ValueError, match="finite"):
        check_forecasts([0.5, np.nan], 2)
    with pytest.raises(ValueError, match="expected 2"):
        check_forecasts([0.5], 2)
    with pytest.raises(ValueError, match="stake_norm"):
        find_dutch_book(AND_NOT, [0.5, 0.5], "l3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm"):
        coherence_distance(AND_NOT, [0.5, 0.5], "l3")  # type: ignore[arg-type]


def test_audit_wrapper_reports_all_distances():
    a = audit(["A", "~A"], [0.6, 0.6])
    assert not a.coherent
    assert a.distances["linf"] == pytest.approx(0.1)
    assert a.distances["l1"] == pytest.approx(0.2)
    assert a.distances["l2"] == pytest.approx(0.1 * 2**0.5)
    assert a.labels == ("A", "¬A")
    with pytest.raises(ValueError, match="labels"):
        audit(["A"], [0.5], labels=["x", "y"])


def test_equivalent_formulas_with_different_prices_form_a_book():
    a = audit(["A", "~~A"], [0.3, 0.5])
    assert a.book.profit == pytest.approx(0.1)  # contradiction between identical events

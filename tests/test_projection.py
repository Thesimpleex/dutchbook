import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize, minimize_scalar

from conftest import random_truth_matrix
from dutchbook import (
    EnumeratedVertices,
    brier_scores,
    dominance_certificate,
    euclidean_projection,
    is_coherent,
    kl_divergence,
    kl_projection,
    log_scores,
    repair,
    world_space,
)


def brute_force_projection(W: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact Euclidean projection by enumerating all supports and solving each KKT system."""
    k = W.shape[0]
    best: tuple[float, np.ndarray] | None = None
    for r in range(1, k + 1):
        for support in itertools.combinations(range(k), r):
            V = W[list(support)]
            kkt = np.zeros((r + 1, r + 1))
            kkt[:r, :r] = 2 * V @ V.T
            kkt[:r, r] = 1.0
            kkt[r, :r] = 1.0
            rhs = np.concatenate([2 * V @ q, [1.0]])
            lam = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:r]
            if (lam >= -1e-12).all() and abs(lam.sum() - 1) < 1e-9:
                p = V.T @ np.clip(lam, 0, None)
                d = float(np.sum((p - q) ** 2))
                if best is None or d < best[0] - 1e-15:
                    best = (d, p)
    assert best is not None
    return best[1], best[0]


def test_matches_brute_force_qp_on_small_cases(rng):
    for _ in range(80):
        W = random_truth_matrix(rng, int(rng.integers(1, 4)), int(rng.integers(2, 6)))
        if W.shape[0] > 7:
            continue
        q = rng.random(W.shape[1])
        p, d = brute_force_projection(W, q)
        proj = euclidean_projection(W, q)
        np.testing.assert_allclose(proj.point, p, atol=1e-9)
        assert np.sum((proj.point - q) ** 2) == pytest.approx(d, abs=1e-12)


def test_kkt_and_duality_gap(rng):
    for _ in range(60):
        W = random_truth_matrix(rng, 3, int(rng.integers(2, 7)))
        q = rng.random(W.shape[1])
        proj = euclidean_projection(W, q)
        # variational inequality: <p - q, w - p> >= 0 for every world w
        assert (W @ (proj.point - q) - (proj.point - q) @ proj.point).min() >= -1e-10
        assert proj.gap <= 1e-12
        # the projected point is a convex combination of its active worlds
        np.testing.assert_allclose(proj.weights @ proj.vertices, proj.point, atol=1e-12)
        assert proj.weights.sum() == pytest.approx(1.0)
        assert (proj.weights >= 0).all()
        # and is coherent according to the independent LP test
        assert is_coherent(W, proj.point, tol=1e-9)


def test_projection_of_a_coherent_point_is_the_point(rng):
    W = random_truth_matrix(rng, 3, 5)
    q = W.T @ rng.dirichlet(np.ones(W.shape[0]))
    proj = euclidean_projection(W, q)
    np.testing.assert_allclose(proj.point, q, atol=1e-9)
    again = euclidean_projection(W, euclidean_projection(W, rng.random(5)).point)
    assert again.gap <= 1e-12


def test_a_and_not_a_projection():
    W = world_space(["A", "~A"]).W
    np.testing.assert_allclose(euclidean_projection(W, [0.6, 0.6]).point, [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(euclidean_projection(W, [0.9, 0.5]).point, [0.7, 0.3], atol=1e-12)


def test_oracle_interface_gives_the_same_projection(rng):
    W = random_truth_matrix(rng, 3, 5)
    q = rng.random(5)
    calls = []

    def oracle(direction):
        calls.append(1)
        return W[int(np.argmin(W @ direction))]

    via_oracle = euclidean_projection(oracle, q)
    direct = euclidean_projection(EnumeratedVertices(W), q)
    np.testing.assert_allclose(via_oracle.point, direct.point, atol=1e-12)
    assert len(calls) == via_oracle.iterations


def test_dense_weights_round_trip(rng):
    W = random_truth_matrix(rng, 3, 4)
    q = rng.random(4)
    proj = euclidean_projection(W, q)
    dense = proj.dense_weights(W)
    assert dense.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(W.T @ dense, proj.point, atol=1e-12)


def test_pythagorean_inequality_hard_invariant_on_random_instances(rng):
    worst = np.inf
    for _ in range(300):
        W = random_truth_matrix(rng, int(rng.integers(1, 5)), int(rng.integers(2, 8)))
        q = rng.random(W.shape[1])
        proj = euclidean_projection(W, q)
        cert = dominance_certificate(W, q, proj.point, "brier", proj.gap)
        assert cert.holds
        worst = min(worst, cert.margin.min())
        # per world: Brier(q, w) - Brier(p, w) >= ||p - q||^2
        improvement = brier_scores(W, q) - brier_scores(W, proj.point)
        assert (improvement >= np.sum((proj.point - q) ** 2) - 1e-12).all()
    assert worst >= -1e-12


def test_dominance_is_strict_iff_incoherent(rng):
    W = random_truth_matrix(rng, 3, 4)
    q = rng.random(4)
    proj = euclidean_projection(W, q)
    cert = dominance_certificate(W, q, proj.point, "brier", proj.gap)
    assert cert.strictly_dominates == (not is_coherent(W, q))
    assert cert.min_improvement >= cert.divergence - 1e-12
    coherent = W.T @ rng.dirichlet(np.ones(W.shape[0]))
    c2 = dominance_certificate(W, coherent, coherent, "brier", 0.0)
    assert not c2.strictly_dominates
    assert c2.divergence == 0.0


def test_dominance_check_has_teeth(rng):
    """A coherent point that is not the projection need not dominate; the check notices."""
    W = world_space(["A", "~A"]).W
    q = np.array([0.9, 0.9])
    not_projection = np.array([1.0, 0.0])  # a vertex: coherent but far
    cert = dominance_certificate(W, q, not_projection, "brier")
    assert not cert.holds
    assert cert.improvement.min() < 0  # worse in the world where A is false


def test_dominance_holds_for_the_whole_vector_not_for_sub_blocks():
    W = world_space(["A", "~A"]).W
    q = np.array([0.9, 0.9])
    proj = euclidean_projection(W, q)
    assert (brier_scores(W, q) - brier_scores(W, proj.point) > 0).all()
    # the first coordinate alone gets worse when A is true: 0.1^2 -> 0.5^2
    world_a_true = 0
    before = (q[0] - W[world_a_true, 0]) ** 2
    after = (proj.point[0] - W[world_a_true, 0]) ** 2
    assert after > before


def test_kl_projection_a_and_not_a():
    W = world_space(["A", "~A"]).W
    proj = kl_projection(W, [0.6, 0.6])
    np.testing.assert_allclose(proj.point, [0.5, 0.5], atol=1e-9)
    # asymmetric case has a closed form: minimise KL(p||q1)+KL(1-p||q2)
    proj = kl_projection(W, [0.8, 0.5])
    target = np.array([0.8, 0.5])
    reference = minimize_scalar(
        lambda x: kl_divergence(np.array([x, 1 - x]), target),
        bounds=(1e-9, 1 - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert proj.point[0] == pytest.approx(reference.x, abs=1e-6)
    assert proj.point[1] == pytest.approx(1 - reference.x, abs=1e-6)


def slsqp_kl_value(W: np.ndarray, q: np.ndarray) -> float:
    """Minimum of the KL divergence over mixtures of worlds by a generic nonlinear solver."""
    k = W.shape[0]
    res = minimize(
        lambda lam: kl_divergence(np.clip(W.T @ lam, 1e-12, 1 - 1e-12), q),
        np.full(k, 1 / k),
        bounds=[(0, 1)] * k,
        constraints=[{"type": "eq", "fun": lambda lam: lam.sum() - 1}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    return float(res.fun)


def test_kl_projection_against_nonlinear_solver(rng):
    for _ in range(40):
        W = random_truth_matrix(rng, 3, int(rng.integers(2, 6)))
        q = rng.uniform(0.02, 0.98, W.shape[1])
        proj = kl_projection(W, q)
        ours = kl_divergence(proj.point, proj.q)
        assert ours <= slsqp_kl_value(W, proj.q) + 1e-8  # never worse than the generic solver
        # first-order optimality over every world: <grad f(p), w - p> >= 0 (convex problem)
        p, qc = np.clip(proj.point, 1e-12, 1 - 1e-12), proj.q
        grad = np.log(p * (1 - qc) / ((1 - p) * qc))
        assert (W @ grad - grad @ proj.point).min() >= -1e-8
        assert proj.gap <= 1e-9
        assert is_coherent(W, proj.point, tol=1e-9)


def test_kl_pythagorean_inequality_in_every_world(rng):
    for _ in range(150):
        W = random_truth_matrix(rng, int(rng.integers(1, 4)), int(rng.integers(2, 7)))
        q = rng.uniform(0.0, 1.0, W.shape[1])
        proj = kl_projection(W, q)
        cert = dominance_certificate(W, proj.q, proj.point, "log", proj.gap)
        assert cert.holds
        improvement = log_scores(W, proj.q) - log_scores(W, proj.point)
        assert (improvement >= kl_divergence(proj.point, proj.q) - 1e-8).all()


def test_kl_projection_handles_degenerate_forecasts():
    W = world_space(["A", "~A", "A | ~A"]).W
    proj = kl_projection(W, [0.0, 1.0, 0.5])  # exact 0 and 1 are clipped to [eps, 1-eps]
    assert np.isfinite(proj.point).all()
    assert proj.point[2] == pytest.approx(1.0)  # the tautology is certain in every world
    assert proj.q.min() > 0
    assert proj.q.max() < 1


def test_repair_wrapper_certificates():
    rep = repair(["A", "~A"], [0.6, 0.6])
    assert rep.certificate.holds
    np.testing.assert_allclose(rep.repaired, [0.5, 0.5], atol=1e-12)
    assert rep.certificate.divergence == pytest.approx(0.02)
    np.testing.assert_allclose(rep.certificate.improvement, [0.02, 0.02], atol=1e-12)
    kl = repair(["A", "~A"], [0.6, 0.6], divergence="kl")
    assert kl.certificate.scoring == "log"
    assert kl.certificate.holds
    with pytest.raises(ValueError, match="divergence"):
        repair(["A"], [0.5], divergence="hellinger")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scoring"):
        dominance_certificate(np.eye(2), [0.5, 0.5], [0.5, 0.5], "spherical")  # type: ignore[arg-type]


def test_projection_input_validation():
    W = world_space(["A", "~A"]).W
    with pytest.raises(ValueError, match="1-D"):
        euclidean_projection(W, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="lie in"):
        euclidean_projection(W, [0.5, 1.5])


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_property_projection_invariances(seed):
    rng = np.random.default_rng(seed)
    W = random_truth_matrix(rng, 3, 5)
    q = rng.random(5)
    base = euclidean_projection(W, q).point
    pf, pw = rng.permutation(5), rng.permutation(W.shape[0])
    permuted = euclidean_projection(W[pw][:, pf], q[pf]).point
    np.testing.assert_allclose(permuted, base[pf], atol=1e-9)
    # complement symmetry: projecting 1 - q onto the polytope of negated formulas
    flipped = euclidean_projection(1 - W, 1 - q).point
    np.testing.assert_allclose(flipped, 1 - base, atol=1e-9)
    # a coherent input is unchanged
    coherent = W.T @ rng.dirichlet(np.ones(W.shape[0]))
    np.testing.assert_allclose(euclidean_projection(W, coherent).point, coherent, atol=1e-9)
    # projection is non-expansive (it projects onto a convex set)
    q2 = rng.random(5)
    d_in = np.linalg.norm(q - q2)
    d_out = np.linalg.norm(base - euclidean_projection(W, q2).point)
    assert d_out <= d_in + 1e-9


def test_constant_coordinates_keep_projections_inside_the_unit_interval(rng):
    # tautology and contradiction columns are constant over worlds; float noise must not
    # push a projected value outside [0, 1] (which would make the log score infinite)
    W = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]])
    for _ in range(50):
        q = rng.random(4)
        for proj, scoring in ((euclidean_projection(W, q), "brier"), (kl_projection(W, q), "log")):
            assert proj.point.min() >= 0.0
            assert proj.point.max() <= 1.0
            cert = dominance_certificate(W, proj.q, proj.point, scoring, proj.gap)
            assert np.isfinite(cert.margin).all()
            assert cert.holds

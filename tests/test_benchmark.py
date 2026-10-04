import numpy as np
import pytest

from dutchbook import benchmark as bm
from dutchbook import coherence_distance, world_space
from dutchbook.forecastbench import FORMULAS, read_tuples


def test_tuple_worlds_are_the_four_cells():
    W = bm.tuple_worlds()
    assert W.shape == (4, 6)
    np.testing.assert_array_equal(W, world_space(FORMULAS).W)
    # marginals are the sums of the corresponding cells
    np.testing.assert_array_equal(W[:, 0], W[:, 2] + W[:, 3])
    np.testing.assert_array_equal(W[:, 1], W[:, 2] + W[:, 4])


def test_renormalise_cells_keeps_marginals_and_sums_to_one():
    q = np.array([[0.3, 0.6, 0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]])
    r = bm.renormalise_cells(q)
    np.testing.assert_allclose(r[:, :2], q[:, :2])
    np.testing.assert_allclose(r[:, 2:].sum(axis=1), 1.0)
    np.testing.assert_allclose(r[0, 2:], 0.25)
    np.testing.assert_allclose(r[1, 2:], 0.25)  # all-zero cells fall back to uniform
    np.testing.assert_array_equal(q[0, 2:], 0.4)  # the input is not modified


def test_constant_baselines_have_closed_form_arbitrage():
    # Always 0: cells sum to one forces some cell >= 1/4 and marginals <= t force t >= 1/3;
    # Always 1/2: cells must sum to one so t >= 1/4; Always 1: four cells >= 1 - t, sum 1: 3/4.
    q = np.array([[0.0] * 6, [0.5] * 6, [1.0] * 6])
    out = bm.solve_unique(q, workers=1)
    np.testing.assert_allclose(out[:, 0], [1 / 3, 1 / 4, 3 / 4], atol=1e-9)
    # the exact LP agrees with the generic API
    for row, expected in zip(q, out[:, 0], strict=True):
        assert coherence_distance(bm.tuple_worlds(), row, "linf").value == pytest.approx(expected)


def test_independence_constructed_tuples_are_coherent():
    a, b = 0.35, 0.3
    q = np.array([[a, b, a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)]])
    assert bm.is_independence_constructed(q).all()
    out = bm.solve_unique(q, workers=1)
    assert out[0, 0] < 1e-12
    np.testing.assert_allclose(out[0, 2:8], q[0], atol=1e-12)
    assert not bm.is_independence_constructed(np.array([[0.5, 0.5, 0.4, 0.1, 0.1, 0.4]])).any()


def test_serial_and_parallel_solves_agree(rng):
    q = rng.random((300, 6))
    q = np.vstack([q, q[:50]])  # duplicates exercise the unique/inverse mapping
    serial = bm.solve_unique(q, workers=1)
    parallel = bm.solve_unique(q, workers=2, chunk=70)
    np.testing.assert_allclose(serial, parallel, atol=1e-12)
    np.testing.assert_allclose(serial[300:], serial[:50])


def test_dominance_margins_are_non_negative_on_random_tuples(rng):
    q = rng.random((400, 6))
    out = bm.solve_unique(q, workers=1)
    assert (out[:, 8] >= -1e-9).all()
    assert (out[:, 9] <= 1e-9).all()
    # the Euclidean distance is at least the sup-norm distance
    assert (out[:, 1] >= out[:, 0] - 1e-12).all()


def test_metrics_and_summaries_on_a_synthetic_snapshot(snapshot_tar, tmp_path):
    table = read_tuples(snapshot_tar)
    metrics = bm.compute_metrics(table, workers=1)
    assert metrics.linf.shape == (len(table),)
    rows = {r["model"]: r for r in bm.model_rows(table, metrics)}
    a = rows["Model-A (zero shot)"]
    assert a["base_model"] == "Model-A"
    assert a["variant"] == "zero shot"
    assert a["n_tuples"] == 4
    assert a["n_resolved"] == 3
    assert a["dominance_violations"] == 0
    assert a["brier_projected"] <= a["brier_raw"]
    assert rows["Always 0.5"]["mean_arbitrage"] == pytest.approx(0.25)
    assert rows["Always 0.5"]["kind"] == "reference"
    overall = bm.overall_summary(table, metrics)
    assert overall["n_models"] == 2
    assert overall["n_tuples"] == 3 + 1 + 2
    # cached_metrics matches and writes one file per round
    cache = tmp_path / "cache"
    cached = bm.cached_metrics(table, ["2024-07-21"], cache, "d" * 16, workers=1)
    assert len(list(cache.glob("metrics-*.npz"))) == 1
    mask = table.round_idx == 0
    np.testing.assert_allclose(cached.linf[mask], metrics.linf[mask])
    assert np.isnan(cached.linf[~mask]).all()
    again = bm.cached_metrics(table, ["2024-07-21"], cache, "d" * 16, workers=1)
    np.testing.assert_allclose(again.linf[mask], cached.linf[mask])
    sub = metrics.select(mask)
    assert sub.linf.shape == (int(mask.sum()),)
    metrics.save(tmp_path / "m.npz")
    np.testing.assert_allclose(bm.TupleMetrics.load(tmp_path / "m.npz").linf, metrics.linf)


def test_split_model_name():
    assert bm.split_model_name("GPT-4o (zero shot)") == ("GPT-4o", "zero shot")
    assert bm.split_model_name("Always 0.5") == ("Always 0.5", "")
    assert bm.split_model_name("Model (scratchpad with freeze values)") == (
        "Model",
        "scratchpad with freeze values",
    )


def test_brier_helpers():
    q = np.array([[0.5] * 6])
    o = np.array([[1, 0, 1, 0, 0, 0]], dtype=float)
    assert bm.brier(q, o)[0] == pytest.approx(0.25)
    assert bm.brier_cells(q, o)[0] == pytest.approx(0.25)


def test_cluster_bootstrap_interval_covers_the_mean(rng):
    clusters = np.repeat(np.arange(200), 5)
    values = rng.normal(1.0, 1.0, 1000) + np.repeat(rng.normal(0, 0.5, 200), 5)
    mean, lo, hi = bm.cluster_bootstrap(values, clusters, reps=500, seed=1)
    assert lo < mean < hi
    assert mean == pytest.approx(values.mean())
    assert hi - lo < 0.5
    # weighted ratio statistic
    w = rng.uniform(1, 2, 1000)
    m2, lo2, hi2 = bm.cluster_bootstrap(values, clusters, reps=200, seed=2, weights=w)
    assert m2 == pytest.approx((values * w).sum() / w.sum())
    assert lo2 < m2 < hi2

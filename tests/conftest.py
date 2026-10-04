import numpy as np
import pytest


def random_truth_matrix(rng: np.random.Generator, n_atoms: int, n_formulas: int) -> np.ndarray:
    """Truth matrix of random formulas given as random subsets of 2**n_atoms worlds."""
    worlds = 2**n_atoms
    cols = rng.random((worlds, n_formulas)) < 0.5
    W = np.unique(cols.astype(float), axis=0)
    return W


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261004)


def _rec(ident, direction, forecast, resolved_to, resolution="2024-12-31T00:00:00.000", **extra):
    base = {
        "id": ident,
        "source": "manifold",
        "direction": direction,
        "forecast_due_date": "2024-07-21T00:00:00.000",
        "resolution_date": resolution,
        "resolved_to": resolved_to,
        "resolved": resolved_to is not None,
        "forecast": forecast,
        "imputed": False,
    }
    base.update(extra)
    return base


def forecast_file(model: str, org: str, due: str, forecasts: list[dict]) -> dict:
    return {
        "organization": "ForecastBench",
        "model": model,
        "model_organization": org,
        "forecast_due_date": due,
        "question_set": f"{due}-llm.json",
        "leaderboard_eligible": True,
        "forecasts": forecasts,
    }


def tuple_records(pair, q, outcomes=(1, 0, 1, 0, 0, 0), **flags):
    """Marginal and joint records of one combination tuple (A true, B false by default)."""
    a, b = pair
    cells = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    recs = [_rec(a, [], q[0], outcomes[0]), _rec(b, [], q[1], outcomes[1])]
    for k, d in enumerate(cells):
        recs.append(_rec([a, b], d, q[2 + k], outcomes[2 + k], **flags))
    return recs


@pytest.fixture
def snapshot_tar(tmp_path):
    """A miniature snapshot: two rounds with combinations (one model each plus a baseline)
    and one later round without combinations, with the data-quality problems of the real one."""
    import io
    import json
    import tarfile

    files: dict[str, dict] = {}
    good = tuple_records(("a1", "b1"), [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    incoh = tuple_records(("a2", "b2"), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], outcomes=(0, 1, 0, 0, 1, 0))
    unresolved = tuple_records(("a3", "b3"), [0.3, 0.3, 0.1, 0.2, 0.2, 0.5], outcomes=(None,) * 6)
    imputed_cell = tuple_records(("a4", "b4"), [0.3, 0.3, 0.1, 0.2, 0.2, 0.5])
    imputed_cell[3]["imputed"] = True
    no_marginal = tuple_records(("a5", "b5"), [0.3, 0.3, 0.1, 0.2, 0.2, 0.5])[2:]
    three_cells = tuple_records(("a6", "b6"), [0.3, 0.3, 0.1, 0.2, 0.2, 0.5])[:-1]
    bad_forecast = tuple_records(("a7", "b7"), [0.3, 0.3, 0.1, 1.2, 0.2, 0.5])
    records = [*good, *incoh, *unresolved, *imputed_cell, *no_marginal, *three_cells, *bad_forecast]
    files["2024-07-21/m1.json"] = forecast_file(
        "Model-A (zero shot)", "Acme", "2024-07-21", records
    )
    files["2024-07-21/m2.json"] = forecast_file(
        "Model-B (zero shot)", "Beta", "2024-07-21", good + incoh
    )
    files["2024-07-21/base.json"] = forecast_file(
        "Always 0.5", "ForecastBench", "2024-07-21", tuple_records(("a1", "b1"), [0.5] * 6)
    )
    files["2025-03-02/m1.json"] = forecast_file(
        "Model-A (zero shot)",
        "Acme",
        "2025-03-02",
        tuple_records(("c1", "d1"), [0.2, 0.7, 0.1, 0.1, 0.6, 0.2]),
    )
    files["2025-10-26/m1.json"] = forecast_file(
        "Model-A (zero shot)", "Acme", "2025-10-26", [_rec("z1", [], 0.5, 1)]
    )
    path = tmp_path / "processed_forecast_sets.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, doc in files.items():
            blob = json.dumps(doc).encode()
            info = tarfile.TarInfo(f"forecastbench-processed-forecast-sets/{name}")
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    return path

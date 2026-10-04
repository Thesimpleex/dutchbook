import json

import numpy as np
import pytest

from dutchbook import forecastbench as fb


def test_read_tuples_extracts_complete_tuples_and_counts_the_rest(snapshot_tar):
    table = fb.read_tuples(snapshot_tar)
    s = table.stats
    assert s["files"] == 5
    assert s["files_with_combinations"] == 4  # the 2025-10-26 file has no combination questions
    assert table.rounds == ("2024-07-21", "2025-03-02")
    # model A: good, incoherent, unresolved in round 1; one more in round 2; model B: two; base: one
    assert len(table) == 3 + 1 + 2 + 1
    assert s["dropped_imputed_cells"] == 1
    assert s["dropped_missing_marginals"] == 1
    assert s["dropped_incomplete_cells"] == 1
    assert s["dropped_invalid_forecast"] == 1
    assert table.q.shape == (7, 6)
    kinds = {m[1]: m[2] for m in table.models}
    assert kinds["Always 0.5"] == "reference"
    assert kinds["Model-A (zero shot)"] == "model"


def test_tuple_contents_and_resolution(snapshot_tar):
    table = fb.read_tuples(snapshot_tar)
    a = next(i for i, m in enumerate(table.models) if m[1] == "Model-A (zero shot)")
    rows = np.where((table.model_idx == a) & (table.round_idx == 0))[0]
    first = rows[0]
    np.testing.assert_allclose(table.q[first], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    np.testing.assert_array_equal(table.outcome[first], [1, 0, 1, 0, 0, 0])
    assert table.pair[first] == "a1|b1"
    assert table.resolution_date[first] == "2024-12-31"
    assert table.source[first] == "manifold"
    assert table.resolved[first]
    unresolved = rows[2]
    assert not table.resolved[unresolved]
    assert np.isnan(table.outcome[unresolved]).all()


def test_table_round_trips_through_the_cache(snapshot_tar, tmp_path):
    table = fb.read_tuples(snapshot_tar)
    path = tmp_path / "t.npz"
    table.save(path)
    back = fb.TupleTable.load(path)
    assert back.rounds == table.rounds
    assert back.models == table.models
    np.testing.assert_array_equal(back.q, table.q)
    np.testing.assert_array_equal(back.pair, table.pair)
    assert back.stats == table.stats
    sub = table.select(table.round_idx == 1)
    assert len(sub) == 1


def test_load_tuples_caches_by_snapshot_digest(snapshot_tar, tmp_path):
    raw = snapshot_tar.parent
    cache = tmp_path / "cache"
    first = fb.load_tuples(raw, cache)
    files = list(cache.glob("tuples-*.npz"))
    assert len(files) == 1
    second = fb.load_tuples(raw, cache)
    np.testing.assert_array_equal(first.q, second.q)
    with pytest.raises(FileNotFoundError, match="download_snapshot"):
        fb.load_tuples(tmp_path / "missing", cache)


class _Response:
    def __init__(self, body: bytes, headers: dict[str, str]):
        self._body, self.headers, self._pos = body, headers, 0

    def read(self, n: int = -1) -> bytes:
        chunk = self._body[self._pos :] if n < 0 else self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_pins_the_snapshot_and_skips_unchanged(tmp_path, monkeypatch):
    body = b"x" * 3000
    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.get_method())
        headers = {
            "ETag": '"abc"',
            "Last-Modified": "Sat, 03 Oct 2026 07:03:22 GMT",
            "Content-Length": str(len(body)),
        }
        return _Response(body if request.get_method() == "GET" else b"", headers)

    monkeypatch.setattr(fb.urllib.request, "urlopen", fake_urlopen)
    seen = []
    manifest = fb.download_snapshot(
        tmp_path, progress=lambda done, total: seen.append((done, total))
    )
    assert calls == ["HEAD", "GET"]
    assert manifest["size"] == 3000
    assert manifest["etag"] == '"abc"'
    assert len(manifest["sha256"]) == 64
    assert manifest["license"] == "CC BY-SA 4.0"
    assert seen[-1] == (3000, 3000)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert fb.snapshot_digest(tmp_path) == manifest["sha256"]
    # unchanged remote: only a HEAD request
    calls.clear()
    again = fb.download_snapshot(tmp_path)
    assert calls == ["HEAD"]
    assert again == manifest
    # forcing downloads again
    fb.download_snapshot(tmp_path, force=True)
    assert calls == ["HEAD", "HEAD", "GET"]


def test_incomplete_download_is_rejected(tmp_path, monkeypatch):
    def fake_urlopen(request, timeout=0):
        headers = {"ETag": '"abc"', "Content-Length": "5000"}
        return _Response(b"y" * 100 if request.get_method() == "GET" else b"", headers)

    monkeypatch.setattr(fb.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(OSError, match="incomplete"):
        fb.download_snapshot(tmp_path)
    assert not (tmp_path / fb.SNAPSHOT_NAME).exists()


def test_question_texts_from_a_question_set(tmp_path, monkeypatch):
    doc = {
        "questions": [
            {
                "id": ["a1", "b1"],
                "question": "combo",
                "combination_of": [
                    {"id": "a1", "question": "Will A happen?"},
                    {"id": "b1", "question": "Will B happen?"},
                ],
            },
            {"id": "z", "question": "Single", "combination_of": "N/A"},
        ]
    }
    (tmp_path / "2024-07-21-llm.json").write_text(json.dumps(doc))
    assert fb.question_texts("2024-07-21", "a1|b1", tmp_path) == [
        "Will A happen?",
        "Will B happen?",
    ]
    assert fb.question_texts("2024-07-21", "a1|zz", tmp_path) == ["Will A happen?", "zz"]
    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append(request.full_url)
        return _Response(json.dumps(doc).encode(), {})

    monkeypatch.setattr(fb.urllib.request, "urlopen", fake_urlopen)
    fb.question_texts("2025-03-02", "a1|b1", tmp_path)
    assert calls == [fb.QUESTION_SET_URL.format(round="2025-03-02")]

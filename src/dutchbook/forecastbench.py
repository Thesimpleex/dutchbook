"""ForecastBench data access: runtime download, pinning and extraction of combination tuples.

ForecastBench (Karger et al., ICLR 2025) published forecasts of language models for
*combination questions*: pairs of binary questions (A, B) for which the four joint
outcomes (A and B, A and not B, not A and B, neither) were forecast. Together with the
two marginal forecasts this gives six numbers over two atoms that must satisfy the
probability calculus. This module downloads the processed forecast sets (CC BY-SA 4.0),
records the snapshot, and extracts every complete tuple without ever writing raw data
into the repository.

Elicitation caveat. ForecastBench asked for each joint outcome in a *separate prompt*
(``combo_index`` selecting one of four prompts in ``src/helpers/model_eval.py`` of
github.com/forecastingresearch/forecastbench; combination questions were removed again
in commits 83c0661 and 20ea01f, issue #149, because they were not being scored). Part of
the incoherence measured on these tuples is therefore an artefact of the elicitation
protocol, not a property of a model's beliefs.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_URL = (
    "https://www.forecastbench.org/assets/data/processed-forecast-sets/"
    "processed_forecast_sets.tar.gz"
)
QUESTION_SET_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/"
    "datasets/question_sets/{round}-llm.json"
)
SNAPSHOT_NAME = "processed_forecast_sets.tar.gz"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CACHE_DIR = Path("data/cache")

FORMULAS: tuple[str, ...] = ("A", "B", "A & B", "A & ~B", "~A & B", "~A & ~B")
"""Formulas of a combination tuple, in the column order used throughout."""

CELL_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
"""ForecastBench ``direction`` of the four joint cells: +1 means the question happens."""

_COMBO_MARKER = re.compile(rb'"id":\s*\[')
_USER_AGENT = "dutchbook (https://github.com/Thesimpleex/dutchbook)"


@dataclass(frozen=True)
class TupleTable:
    """All complete combination tuples of a snapshot.

    Attributes
    ----------
    rounds : tuple of str
        Forecast due dates (question-set rounds) containing combination questions.
    models : tuple of (str, str, str)
        ``(model_organization, model, kind)`` with ``kind`` either ``"model"`` or
        ``"reference"`` (ForecastBench's own baselines and crowd aggregates).
    round_idx, model_idx : ndarray of int
        Per tuple, indices into ``rounds`` and ``models``.
    q : ndarray, shape (n, 6)
        Forecasts for ``A, B, A∧B, A∧¬B, ¬A∧B, ¬A∧¬B``.
    outcome : ndarray, shape (n, 6)
        Resolution of the six events (0/1), ``nan`` where not (yet) resolved.
    pair : ndarray of str
        ``"idA|idB"`` question pair identifier.
    resolution_date : ndarray of str
    source : ndarray of str
        Question source (market platform or data source) of the pair.
    stats : dict
        Counters of what was read and what was dropped.
    """

    rounds: tuple[str, ...]
    models: tuple[tuple[str, str, str], ...]
    round_idx: np.ndarray
    model_idx: np.ndarray
    q: np.ndarray
    outcome: np.ndarray
    pair: np.ndarray
    resolution_date: np.ndarray
    source: np.ndarray
    stats: dict[str, int]

    def __len__(self) -> int:
        return int(self.q.shape[0])

    @property
    def resolved(self) -> np.ndarray:
        """Mask of tuples whose six events are all resolved to 0 or 1."""
        return ~np.isnan(self.outcome).any(axis=1)

    def select(self, mask: np.ndarray) -> TupleTable:
        return TupleTable(
            self.rounds,
            self.models,
            self.round_idx[mask],
            self.model_idx[mask],
            self.q[mask],
            self.outcome[mask],
            self.pair[mask],
            self.resolution_date[mask],
            self.source[mask],
            self.stats,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            rounds=np.array(self.rounds),
            models=np.array(self.models),
            round_idx=self.round_idx,
            model_idx=self.model_idx,
            q=self.q,
            outcome=self.outcome,
            pair=self.pair,
            resolution_date=self.resolution_date,
            source=self.source,
            stats=np.array(json.dumps(self.stats)),
        )

    @classmethod
    def load(cls, path: str | Path) -> TupleTable:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                tuple(str(r) for r in z["rounds"]),
                tuple((str(a), str(b), str(c)) for a, b, c in z["models"]),
                z["round_idx"],
                z["model_idx"],
                z["q"],
                z["outcome"],
                z["pair"],
                z["resolution_date"],
                z["source"],
                json.loads(str(z["stats"])),
            )


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, method=method, headers={"User-Agent": _USER_AGENT})


def snapshot_manifest_path(raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    return Path(raw_dir) / "manifest.json"


def read_manifest(raw_dir: str | Path = DEFAULT_RAW_DIR) -> dict[str, Any]:
    """Pinned description of the downloaded snapshot (empty dict if none)."""
    path = snapshot_manifest_path(raw_dir)
    return json.loads(path.read_text()) if path.exists() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_digest(raw_dir: str | Path = DEFAULT_RAW_DIR) -> str:
    """SHA-256 of the downloaded snapshot (from the manifest if present, else computed)."""
    manifest = read_manifest(raw_dir)
    if manifest.get("sha256"):
        return str(manifest["sha256"])
    return _sha256(Path(raw_dir) / SNAPSHOT_NAME)


def download_snapshot(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Download the processed forecast sets into ``raw_dir`` and pin the snapshot.

    The remote ETag, last-modified time and size are compared with the manifest of an
    earlier download; the file is only fetched again when they differ (or ``force``).
    The manifest records the URL, download date, ETag, size and SHA-256 and is returned.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / SNAPSHOT_NAME
    manifest = read_manifest(raw)
    with urllib.request.urlopen(_request(SNAPSHOT_URL, "HEAD"), timeout=60) as head:
        etag = head.headers.get("ETag", "")
        last_modified = head.headers.get("Last-Modified", "")
        size = int(head.headers.get("Content-Length", "0"))
    if (
        not force
        and target.exists()
        and manifest.get("etag") == etag
        and manifest.get("size") == size
        and target.stat().st_size == size
    ):
        return manifest
    tmp = target.with_suffix(".part")
    done = 0
    with urllib.request.urlopen(_request(SNAPSHOT_URL), timeout=120) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if progress:
                progress(done, size)
    if size and done != size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"incomplete download: {done} of {size} bytes")
    tmp.replace(target)
    manifest = {
        "url": SNAPSHOT_URL,
        "file": SNAPSHOT_NAME,
        "downloaded": datetime.now(UTC).date().isoformat(),
        "etag": etag,
        "last_modified": last_modified,
        "size": done,
        "sha256": _sha256(target),
        "license": "CC BY-SA 4.0",
    }
    snapshot_manifest_path(raw).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def download_question_set(round_date: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    """Fetch the LLM question set of a round (question texts) into ``raw_dir``."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / f"{round_date}-llm.json"
    if not target.exists():
        with urllib.request.urlopen(
            _request(QUESTION_SET_URL.format(round=round_date)), timeout=120
        ) as resp:
            data = resp.read()
        target.write_bytes(data)
    return target


def question_texts(round_date: str, pair: str, raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[str]:
    """Texts of the two questions of a pair (downloads the round's question set if needed)."""
    path = download_question_set(round_date, raw_dir)
    data = json.loads(path.read_text())
    ids = pair.split("|")
    texts: dict[str, str] = {}
    for question in data["questions"]:
        combo = question.get("combination_of")
        if isinstance(combo, list):
            for sub in combo:
                texts[str(sub["id"])] = str(sub["question"])
        elif question.get("id") is not None:
            texts[str(question["id"])] = str(question["question"])
    return [texts.get(i, i) for i in ids]


def _kind(model_organization: str) -> str:
    return "reference" if model_organization == "ForecastBench" else "model"


def _outcome(rec: dict[str, Any]) -> float:
    if not rec.get("resolved"):
        return float("nan")
    value = rec.get("resolved_to")
    return float(value) if value in (0, 1) else float("nan")


def _file_tuples(doc: dict[str, Any], stats: dict[str, int]) -> list[tuple[Any, ...]]:
    """Complete tuples of one forecast file."""
    singles: dict[tuple[str, str], dict[str, Any]] = {}
    groups: dict[tuple[tuple[str, str], str], dict[tuple[int, int], dict[str, Any]]] = {}
    for rec in doc["forecasts"]:
        ident = rec["id"]
        if isinstance(ident, list):
            key = ((str(ident[0]), str(ident[1])), rec.get("resolution_date") or "")
            groups.setdefault(key, {})[tuple(rec["direction"])] = rec
        elif not rec.get("imputed"):
            singles[(str(ident), rec.get("resolution_date") or "")] = rec
        else:
            stats["imputed_marginals"] += 1
    out = []
    for ((id_a, id_b), res), cells in groups.items():
        stats["combo_groups"] += 1
        if any(d not in cells for d in CELL_DIRECTIONS):
            stats["dropped_incomplete_cells"] += 1
            continue
        if any(cells[d].get("imputed") for d in CELL_DIRECTIONS):
            stats["dropped_imputed_cells"] += 1
            continue
        a, b = singles.get((id_a, res)), singles.get((id_b, res))
        if a is None or b is None:
            stats["dropped_missing_marginals"] += 1
            continue
        recs = [a, b, *(cells[d] for d in CELL_DIRECTIONS)]
        q = np.array([r.get("forecast") for r in recs], dtype=float)
        if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
            stats["dropped_invalid_forecast"] += 1
            continue
        out.append(
            (
                q,
                np.array([_outcome(r) for r in recs]),
                f"{id_a}|{id_b}",
                res[:10],
                str(cells[CELL_DIRECTIONS[0]].get("source", "")),
            )
        )
    return out


def read_tuples(
    tar_path: str | Path,
    progress: Callable[[str], None] | None = None,
) -> TupleTable:
    """Stream the snapshot tarball and extract every complete combination tuple.

    A tuple needs all four joint-cell forecasts and both marginal forecasts for the same
    pair and resolution date, none of them imputed by the benchmark. Files without
    combination questions (the rounds after 2025-08-31) are skipped without parsing.
    """
    stats = {
        "files": 0,
        "files_with_combinations": 0,
        "combo_groups": 0,
        "imputed_marginals": 0,
        "dropped_incomplete_cells": 0,
        "dropped_imputed_cells": 0,
        "dropped_missing_marginals": 0,
        "dropped_invalid_forecast": 0,
    }
    rounds: dict[str, int] = {}
    models: dict[tuple[str, str, str], int] = {}
    cols: dict[str, list[Any]] = {
        k: [] for k in ("round", "model", "q", "outcome", "pair", "res", "source")
    }
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            stats["files"] += 1
            handle = tar.extractfile(member)
            if handle is None:
                continue
            raw = handle.read()
            if _COMBO_MARKER.search(raw) is None:
                continue
            doc = json.loads(raw)
            stats["files_with_combinations"] += 1
            round_date = str(doc.get("forecast_due_date") or member.name.split("/")[-2])[:10]
            org = str(doc.get("model_organization", ""))
            key = (org, str(doc.get("model", "")), _kind(org))
            r_i = rounds.setdefault(round_date, len(rounds))
            m_i = models.setdefault(key, len(models))
            for q, outcome, pair, res, source in _file_tuples(doc, stats):
                cols["round"].append(r_i)
                cols["model"].append(m_i)
                cols["q"].append(q)
                cols["outcome"].append(outcome)
                cols["pair"].append(pair)
                cols["res"].append(res)
                cols["source"].append(source)
            if progress:
                progress(f"{round_date} {key[1]}")
    order = sorted(rounds, key=lambda r: r)
    remap = np.array([order.index(r) for r in rounds], dtype=np.int16)
    n = len(cols["q"])
    stats["tuples"] = n
    return TupleTable(
        rounds=tuple(order),
        models=tuple(models),
        round_idx=remap[np.array(cols["round"], dtype=np.int64)] if n else np.zeros(0, np.int16),
        model_idx=np.array(cols["model"], dtype=np.int32),
        q=np.array(cols["q"], dtype=float).reshape(n, 6),
        outcome=np.array(cols["outcome"], dtype=float).reshape(n, 6),
        pair=np.array(cols["pair"], dtype=str),
        resolution_date=np.array(cols["res"], dtype=str),
        source=np.array(cols["source"], dtype=str),
        stats=stats,
    )


def load_tuples(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    progress: Callable[[str], None] | None = None,
) -> TupleTable:
    """Extract the tuples of the downloaded snapshot, caching the result under ``cache_dir``."""
    raw_path = Path(raw_dir) / SNAPSHOT_NAME
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found; run download_snapshot() first")
    digest = snapshot_digest(raw_dir)
    cache = Path(cache_dir) / f"tuples-{digest[:16]}.npz"
    if cache.exists():
        return TupleTable.load(cache)
    table = read_tuples(raw_path, progress)
    table.save(cache)
    return table

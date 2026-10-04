"""Input files: formula specifications (TOML) and forecast tables (CSV).

A specification names the formulas that forecasts refer to::

    title = "Linda"

    [atoms]                      # optional human-readable descriptions
    F = "Linda is a feminist"
    B = "Linda is a bank teller"

    [formulas]
    bank_teller = "B"
    feminist_bank_teller = "F & B"

A forecast table has the columns ``id`` and ``forecast`` and, optionally, ``group`` to
audit several forecast sets (forecasters, questions, dates) from one file::

    group,id,forecast
    typical,bank_teller,0.05
    typical,feminist_bank_teller,0.30
"""

from __future__ import annotations

import csv
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dutchbook.logic import Formula, parse

DEFAULT_GROUP = "forecasts"


@dataclass(frozen=True)
class Spec:
    """Named formulas, with optional descriptions of the atoms."""

    ids: tuple[str, ...]
    formulas: tuple[Formula, ...]
    title: str = ""
    atoms: dict[str, str] = field(default_factory=dict)

    def formula(self, ident: str) -> Formula:
        try:
            return self.formulas[self.ids.index(ident)]
        except ValueError:
            raise KeyError(f"unknown formula id {ident!r}") from None


@dataclass(frozen=True)
class ForecastSet:
    """The forecasts of one group, aligned with a subset of the specification."""

    group: str
    ids: tuple[str, ...]
    values: np.ndarray


def load_spec(path: str | Path) -> Spec:
    """Read a TOML specification; every formula string is parsed and validated."""
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc
    raw = data.get("formulas")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: missing or empty [formulas] table")
    ids, formulas = [], []
    for ident, text in raw.items():
        if not isinstance(text, str):
            raise ValueError(f"{path}: formula {ident!r} must be a string")
        try:
            formulas.append(parse(text))
        except ValueError as exc:
            raise ValueError(f"{path}: formula {ident!r}: {exc}") from exc
        ids.append(str(ident))
    atoms = {str(k): str(v) for k, v in (data.get("atoms") or {}).items()}
    return Spec(tuple(ids), tuple(formulas), str(data.get("title", "")), atoms)


def parse_probability(text: str) -> float:
    """``0.3``, ``30%`` or ``.3`` to a float in [0, 1]."""
    s = text.strip()
    value = float(s[:-1]) / 100.0 if s.endswith("%") else float(s)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"probability {text!r} is outside [0, 1]")
    return value


def read_forecasts(path: str | Path, spec: Spec) -> list[ForecastSet]:
    """Read a forecast table and align each group with the specification.

    Only the formulas a group actually forecast are audited, in specification order.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "id" not in header or "forecast" not in header:
            raise ValueError(f"{path}: needs the columns 'id' and 'forecast' (found {header})")
        groups: dict[str, dict[str, float]] = {}
        for line, row in enumerate(reader, start=2):
            row = {k.strip().lower(): (v or "") for k, v in row.items() if k is not None}
            ident = row["id"].strip()
            if ident not in spec.ids:
                raise ValueError(f"{path}:{line}: id {ident!r} is not defined in the specification")
            group = row.get("group", "").strip() or DEFAULT_GROUP
            try:
                value = parse_probability(row["forecast"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
            bucket = groups.setdefault(group, {})
            if ident in bucket:
                raise ValueError(f"{path}:{line}: duplicate forecast for {ident!r} in {group!r}")
            bucket[ident] = value
    if not groups:
        raise ValueError(f"{path}: no forecasts found")
    out = []
    for group, values in groups.items():
        ids = tuple(i for i in spec.ids if i in values)
        out.append(ForecastSet(group, ids, np.array([values[i] for i in ids])))
    return out

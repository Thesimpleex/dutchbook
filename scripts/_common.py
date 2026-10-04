"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results.json"
FIGURES = ROOT / "docs" / "figures"
SCREENSHOTS = ROOT / "docs" / "screenshots"
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"


def load_results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def update_results(section: str, payload: Any) -> None:
    """Replace one top-level section of ``docs/results.json`` and refresh the metadata."""
    data = load_results()
    data[section] = payload
    data["meta"] = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: metadata.version(name) for name in ("dutchbook", "numpy", "scipy", "rich")
        },
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

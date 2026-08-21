"""Versioned result envelopes and JSON serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


RESULT_SCHEMA_VERSION = 1


def artifact_matches(path: str | Path, *, experiment: str, seed: int,
                     configuration_hash: str) -> bool:
    """Return true only for a complete artifact from the same run definition."""
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        document.get("experiment") == experiment
        and document.get("seed") == seed
        and document.get("configuration_hash") == configuration_hash
        and document.get("completed") is True
    )


def result_envelope(
    *,
    experiment: str,
    method: str,
    seed: int,
    payload: Mapping[str, Any],
    assumptions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "method_assumptions": dict(assumptions or {}),
        **dict(payload),
    }


def write_json_result(result: Mapping[str, Any], destination: str | Path) -> Path:
    """Atomically write a result mapping without depending on any method."""

    serialized = dict(result)
    serialized.setdefault("schema_version", RESULT_SCHEMA_VERSION)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path

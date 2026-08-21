"""Generic NumPy checkpoints over the public learner-state contract."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from continual_core.protocols import ProtocolError
from continual_core.state import persistent_state, restore, transient_state


CHECKPOINT_SCHEMA_VERSION = 1
PERSISTENT_PREFIX = "persistent__"
TRANSIENT_PREFIX = "transient__"


def _ensure_transaction_closed(learner: object) -> None:
    if getattr(learner, "_pending_prediction", None) is not None:
        raise ProtocolError("cannot checkpoint between predict and update")


def _qualified_type(learner: object) -> str:
    learner_type = type(learner)
    return f"{learner_type.__module__}.{learner_type.__qualname__}"


def save_checkpoint(learner: object, destination: str | Path) -> Path:
    """Atomically save persistent and transient arrays for any public learner."""

    _ensure_transaction_closed(learner)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.array([CHECKPOINT_SCHEMA_VERSION], dtype=np.int64),
        "learner_type": np.array([_qualified_type(learner)]),
    }
    arrays.update(
        {
            f"{PERSISTENT_PREFIX}{name}": array
            for name, array in persistent_state(learner).items()
        }
    )
    arrays.update(
        {
            f"{TRANSIENT_PREFIX}{name}": array
            for name, array in transient_state(learner).items()
        }
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def _checkpoint_group(
    archive: np.lib.npyio.NpzFile, prefix: str
) -> dict[str, np.ndarray]:
    return {
        name[len(prefix) :]: archive[name]
        for name in archive.files
        if name.startswith(prefix)
    }


def restore_checkpoint(learner: object, source: str | Path) -> None:
    """Restore a checkpoint into a learner with the same type and state keys."""

    _ensure_transaction_closed(learner)
    with np.load(Path(source), allow_pickle=False) as archive:
        version = int(archive["schema_version"][0])
        if version != CHECKPOINT_SCHEMA_VERSION:
            raise ProtocolError(
                f"checkpoint schema {version} is not supported"
            )
        checkpoint_type = str(archive["learner_type"][0])
        expected_type = _qualified_type(learner)
        if checkpoint_type != expected_type:
            raise ProtocolError(
                f"checkpoint learner type {checkpoint_type!r} does not match "
                f"{expected_type!r}"
            )
        restore(
            persistent_state(learner),
            _checkpoint_group(archive, PERSISTENT_PREFIX),
        )
        restore(
            transient_state(learner),
            _checkpoint_group(archive, TRANSIENT_PREFIX),
        )

"""Generic state inspection, locking, and restoration utilities."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from continual_core.protocols import ProtocolError


Array = NDArray[Any]


def _array_mapping(value: object, attribute: str) -> dict[str, Array]:
    state = getattr(value, attribute, None)
    if state is None:
        return {}
    if callable(state):
        state = state()
    if not isinstance(state, Mapping):
        raise ProtocolError(f"{attribute} must be a mapping of NumPy arrays")
    result: dict[str, Array] = {}
    for name, array in state.items():
        if not isinstance(name, str) or not isinstance(array, np.ndarray):
            raise ProtocolError(f"{attribute} must map strings to NumPy arrays")
        result[name] = array
    return result


def persistent_state(value: object) -> dict[str, Array]:
    """Return named persistent arrays without knowing a concrete model type."""

    result = _array_mapping(value, "persistent_state")
    if result:
        return result
    arrays = getattr(value, "persistent_arrays", None)
    if arrays is None:
        raise ProtocolError(
            f"{type(value).__name__} does not expose persistent_state"
        )
    if callable(arrays):
        arrays = arrays()
    if not isinstance(arrays, tuple) or not all(
        isinstance(array, np.ndarray) for array in arrays
    ):
        raise ProtocolError("persistent_arrays must be a tuple of NumPy arrays")
    return {f"array_{index:04d}": array for index, array in enumerate(arrays)}


def transient_state(value: object) -> dict[str, Array]:
    """Return named transient arrays, or an empty mapping for stateless models."""

    return _array_mapping(value, "transient_state")


def state_nbytes(value: object) -> int:
    """Count persistent bytes from the public state contract."""

    return sum(array.nbytes for array in persistent_state(value).values())


def snapshot(arrays: Mapping[str, Array]) -> dict[str, Array]:
    return {name: array.copy() for name, array in arrays.items()}


def restore(arrays: Mapping[str, Array], saved: Mapping[str, Array]) -> None:
    if arrays.keys() != saved.keys():
        raise ProtocolError("state keys changed while restoring a snapshot")
    for name, array in arrays.items():
        source = saved[name]
        if array.shape != source.shape or array.dtype != source.dtype:
            raise ProtocolError(f"state array {name!r} changed shape or dtype")
        np.copyto(array, source)


@contextmanager
def locked_state(value: object) -> Iterator[None]:
    """Prohibit persistent mutation and restore transient state on exit."""

    persistent = persistent_state(value)
    transient = transient_state(value)
    persistent_before = snapshot(persistent)
    transient_before = snapshot(transient)
    try:
        yield
    finally:
        restore(transient, transient_before)
    changed = [
        name
        for name, array in persistent.items()
        if not np.array_equal(array, persistent_before[name], equal_nan=True)
    ]
    if changed:
        raise ProtocolError(
            "locked evaluation modified persistent state: " + ", ".join(changed)
        )

"""Shared validation helpers for independent baseline implementations."""

import numpy as np

from continual_core.protocols import FloatArray


def vector(name: str, value: FloatArray, size: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")

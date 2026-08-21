"""Small validation helpers shared by protocol implementations."""

from __future__ import annotations

import numpy as np

from continual_core.protocols import FloatArray


def positive(name: str, value: float) -> None:
    """Require a finite positive scalar for public component configuration."""

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def vector(name: str, value: FloatArray, size: int) -> FloatArray:
    """Return *value* as a float vector with the required shape."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    return array

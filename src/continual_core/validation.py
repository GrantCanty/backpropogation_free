"""Small validation helpers shared by protocol implementations."""

from __future__ import annotations

import numpy as np

from continual_core.protocols import FloatArray


def vector(name: str, value: FloatArray, size: int) -> FloatArray:
    """Return *value* as a float vector with the required shape."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    return array

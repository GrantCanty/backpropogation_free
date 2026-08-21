"""Types and state checks for predict-before-learn streaming."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StreamEvent:
    """One prequential event from a streaming environment."""

    step: int
    observation: FloatArray
    target: FloatArray
    regime: int
    change_point: bool = False


class ProtocolError(RuntimeError):
    """Raised when predict and learn are called out of order."""

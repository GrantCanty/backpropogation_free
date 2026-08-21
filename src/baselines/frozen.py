"""Frozen linear readout control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class FrozenReadout:
    input_size: int
    output_size: int
    seed: int = 0
    initial_scale: float = 0.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(
            0.0, self.initial_scale, size=(self.output_size, self.input_size)
        )

    def predict(self, features: FloatArray) -> FloatArray:
        return self.weights @ vector("features", features, self.input_size)

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        vector("features", features, self.input_size)
        vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.weights,)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights}

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def diagnostics(self) -> dict[str, object]:
        return {"algorithm": type(self).__name__, "stored_raw_examples": 0}

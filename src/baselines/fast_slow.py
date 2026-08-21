"""Fast/slow baseline controls with explicit consolidation policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.prototype import PrototypeReadout
from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class ProtectedFastSlowReadout:
    input_size: int
    output_size: int
    seed: int = 0
    fast_learning_rate: float = 0.12
    fast_decay: float = 0.995
    fast_scale: float = 0.35
    epsilon: float = 1e-6
    update_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.fast_decay <= 1.0:
            raise ValueError("fast_decay must be in [0, 1]")
        if self.fast_scale < 0.0:
            raise ValueError("fast_scale cannot be negative")
        self.slow_memory = PrototypeReadout(self.input_size, self.output_size)
        self.fast_weights = np.zeros((self.output_size, self.input_size))

    @property
    def weights(self) -> FloatArray:
        return self.slow_memory.centroids + self.fast_scale * self.fast_weights

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        return self.slow_memory.predict(features) + self.fast_scale * (
            self.fast_weights @ features
        )

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        self.fast_weights *= self.fast_decay
        update = self.fast_learning_rate * np.outer(
            target - prediction, features
        ) / (self.epsilon + float(features @ features))
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.fast_weights += update
        self.slow_memory.update(features, target, prediction)

    @property
    def state_nbytes(self) -> int:
        return self.slow_memory.state_nbytes + self.fast_weights.nbytes

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (*self.slow_memory.persistent_arrays, self.fast_weights)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "slow_centroids": self.slow_memory.centroids,
            "slow_counts": self.slow_memory.counts,
            "fast_weights": self.fast_weights,
        }

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def diagnostics(self) -> dict[str, object]:
        return {"algorithm": "protected_fast_slow", "stored_raw_examples": 0}


@dataclass
class FastSlowLMSReadout:
    input_size: int
    output_size: int
    seed: int = 0
    learning_rate: float = 0.15
    fast_decay: float = 0.995
    consolidation_rate: float = 0.002
    epsilon: float = 1e-6
    update_clip: float | None = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.fast_decay <= 1.0:
            raise ValueError("fast_decay must be in [0, 1]")
        if not 0.0 <= self.consolidation_rate <= 1.0:
            raise ValueError("consolidation_rate must be in [0, 1]")
        self.slow_weights = np.zeros((self.output_size, self.input_size))
        self.fast_weights = np.zeros_like(self.slow_weights)

    @property
    def weights(self) -> FloatArray:
        return self.slow_weights + self.fast_weights

    def predict(self, features: FloatArray) -> FloatArray:
        return self.weights @ vector("features", features, self.input_size)

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        self.fast_weights *= self.fast_decay
        update = self.learning_rate * np.outer(
            target - prediction, features
        ) / (self.epsilon + float(features @ features))
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.fast_weights += update
        transfer = self.consolidation_rate * self.fast_weights
        self.slow_weights += transfer
        self.fast_weights -= transfer

    @property
    def state_nbytes(self) -> int:
        return self.slow_weights.nbytes + self.fast_weights.nbytes

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.slow_weights, self.fast_weights)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "slow_weights": self.slow_weights,
            "fast_weights": self.fast_weights,
        }

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def diagnostics(self) -> dict[str, object]:
        return {"algorithm": "fast_slow_lms", "stored_raw_examples": 0}

"""Cumulative per-class prototype baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class PrototypeReadout:
    input_size: int
    output_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.centroids = np.zeros((self.output_size, self.input_size))
        self.counts = np.zeros(self.output_size)

    @property
    def weights(self) -> FloatArray:
        return self.centroids

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        if not np.any(self.counts > 0.0):
            return np.zeros(self.output_size)
        scores = -np.mean(np.square(self.centroids - features), axis=1)
        seen = self.counts > 0.0
        scores[~seen] = float(np.min(scores[seen]) - 1.0)
        return scores

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        label = int(np.argmax(target))
        self.counts[label] += 1.0
        self.centroids[label] += (
            features - self.centroids[label]
        ) / self.counts[label]

    @property
    def state_nbytes(self) -> int:
        return self.centroids.nbytes + self.counts.nbytes

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.centroids, self.counts)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {"centroids": self.centroids, "counts": self.counts}

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def diagnostics(self) -> dict[str, object]:
        return {"algorithm": "class_prototypes", "stored_raw_examples": 0}

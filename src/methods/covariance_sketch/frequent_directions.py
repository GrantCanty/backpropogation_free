"""Frequent-Directions covariance sketch for cumulative ridge regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class FrequentDirectionsRidgeReadout:
    """Approximate cumulative ridge regression with bounded covariance state.

    The input covariance ``X.T @ X`` is represented by a deterministic
    Frequent-Directions sketch while ``X.T @ Y`` remains exact.  No examples,
    gradients, optimizer state, or forgetting factor are used.
    """

    input_size: int
    output_size: int
    sketch_rank: int = 16
    regularization: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        del self.seed  # Interface parity; the algorithm itself is deterministic.
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if not 0 < self.sketch_rank < self.input_size:
            raise ValueError("sketch_rank must be in [1, input_size)")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        self.weights = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.sketch = np.zeros(
            (2 * self.sketch_rank, self.input_size), dtype=np.float64
        )
        self.cross_covariance = np.zeros(
            (self.input_size, self.output_size), dtype=np.float64
        )
        self.filled_rows = np.zeros(1, dtype=np.int64)
        self.sample_count = np.zeros(1, dtype=np.int64)
        self.compression_count = np.zeros(1, dtype=np.int64)
        self.last_condition_number = np.ones(1, dtype=np.float64)

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        return self.weights @ features

    def _compress(self) -> None:
        _, singular_values, right = np.linalg.svd(
            self.sketch, full_matrices=False
        )
        delta = float(singular_values[self.sketch_rank] ** 2)
        shrunk = np.sqrt(
            np.maximum(singular_values[: self.sketch_rank] ** 2 - delta, 0.0)
        )
        self.sketch.fill(0.0)
        self.sketch[: self.sketch_rank] = shrunk[:, None] * right[
            : self.sketch_rank
        ]
        self.filled_rows[0] = self.sketch_rank
        self.compression_count[0] += 1

    def _solve(self) -> None:
        active = self.sketch[: int(self.filled_rows[0])]
        cross = self.cross_covariance
        if len(active) == 0:
            self.weights.fill(0.0)
            return
        reduced = active @ active.T
        reduced.flat[:: len(active) + 1] += self.regularization
        projected_cross = active @ cross
        try:
            correction = np.linalg.solve(reduced, projected_cross)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(
                reduced, projected_cross, rcond=None
            )[0]
        coefficients = (
            cross - active.T @ correction
        ) / self.regularization
        self.weights[:] = coefficients.T
        self.last_condition_number[0] = float(np.linalg.cond(reduced))

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        filled = int(self.filled_rows[0])
        self.sketch[filled] = features
        self.filled_rows[0] += 1
        if int(self.filled_rows[0]) == len(self.sketch):
            self._compress()
        self.cross_covariance += np.outer(features, target)
        self.sample_count[0] += 1
        self._solve()

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.weights,
            self.sketch,
            self.cross_covariance,
            self.filled_rows,
            self.sample_count,
            self.compression_count,
            self.last_condition_number,
        )

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            "sketch": self.sketch,
            "cross_covariance": self.cross_covariance,
            "filled_rows": self.filled_rows,
            "sample_count": self.sample_count,
            "compression_count": self.compression_count,
            "last_condition_number": self.last_condition_number,
        }

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "frequent_directions_ridge",
            "sketch_rank": self.sketch_rank,
            "filled_rows": int(self.filled_rows[0]),
            "compressions": int(self.compression_count[0]),
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "last_reduced_condition_number": float(
                self.last_condition_number[0]
            ),
            "forgetting_factor": None,
            "stored_raw_observations": 0,
        }

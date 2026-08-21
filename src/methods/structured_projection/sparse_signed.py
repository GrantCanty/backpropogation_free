"""Sparse signed frozen features for the projection ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_core.protocols import FloatArray
from continual_core.validation import positive, vector


@dataclass(frozen=True)
class SparseSignedFeatureMap:
    """A fixed sparse signed tanh projection with an explicit bias coordinate.

    The representation intentionally stores coordinates and signs instead of
    a dense weight matrix.  ``transform`` computes each sparse dot product
    directly, so no dense equivalent is constructed even temporarily.
    """

    input_size: int
    hidden_size: int
    fan_in: int
    seed: int = 0
    input_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("all dimensions must be positive")
        if self.fan_in <= 0 or self.fan_in > self.input_size:
            raise ValueError("fan_in must be in [1, input_size]")
        positive("input_scale", self.input_scale)
        rng = np.random.default_rng(self.seed)
        indices = np.empty((self.hidden_size, self.fan_in), dtype=np.int64)
        for row in range(self.hidden_size):
            indices[row] = rng.choice(self.input_size, self.fan_in, replace=False)
        signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=indices.shape)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(
            self,
            "hidden_bias",
            rng.uniform(-self.input_scale, self.input_scale, self.hidden_size),
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size + 1

    @property
    def _weight_scale(self) -> float:
        return self.input_scale / np.sqrt(self.fan_in)

    def transform(self, inputs: FloatArray) -> FloatArray:
        values = vector("inputs", inputs, self.input_size)
        selected = values[self.indices]
        hidden = np.tanh(
            self._weight_scale * np.sum(selected * self.signs, axis=1)
            + self.hidden_bias
        )
        return np.concatenate((hidden, np.ones(1, dtype=np.float64)))

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.indices, self.signs, self.hidden_bias)

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "sparse_signed_frozen_projection",
            "fan_in": self.fan_in,
            "stored_dense_weights": False,
            "stored_raw_examples": 0,
        }

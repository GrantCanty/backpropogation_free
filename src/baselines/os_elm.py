"""Online Sequential Extreme Learning Machine baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines._validation import positive, vector
from continual_core.protocols import FloatArray


@dataclass(frozen=True)
class OSELMFeatureMap:
    """Deterministic frozen random features used by OS-ELM.

    The feature map is public so solver-comparison experiments can materialize
    one representation once and give the exact same coordinates to every
    output learner.  The final coordinate is a constant output bias.
    """

    input_size: int
    hidden_size: int
    seed: int = 0
    input_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("all dimensions must be positive")
        positive("input_scale", self.input_scale)
        rng = np.random.default_rng(self.seed)
        object.__setattr__(
            self,
            "hidden_weights",
            rng.normal(
                0.0,
                self.input_scale / np.sqrt(self.input_size),
                size=(self.hidden_size, self.input_size),
            ),
        )
        object.__setattr__(
            self,
            "hidden_bias",
            rng.uniform(-self.input_scale, self.input_scale, self.hidden_size),
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size + 1

    def transform(self, inputs: FloatArray) -> FloatArray:
        inputs = vector("inputs", inputs, self.input_size)
        hidden = np.tanh(self.hidden_weights @ inputs + self.hidden_bias)
        return np.concatenate((hidden, np.ones(1, dtype=np.float64)))

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.hidden_weights, self.hidden_bias)

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)


@dataclass
class OnlineSequentialELMReadout:
    """Online Sequential ELM with fixed random tanh features and RLS output.

    Classical OS-ELM initializes from a block and then processes samples or
    chunks recursively.  This regularized event-wise form starts from the ridge
    prior, which preserves the repository's predict-before-update contract.
    """

    input_size: int
    output_size: int
    hidden_size: int
    seed: int = 0
    input_scale: float = 0.5
    regularization: float = 1.0

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0 or self.hidden_size <= 0:
            raise ValueError("all dimensions must be positive")
        positive("input_scale", self.input_scale)
        positive("regularization", self.regularization)
        self.feature_map = OSELMFeatureMap(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            seed=self.seed,
            input_scale=self.input_scale,
        )
        # Keep these public aliases for checkpoint and baseline compatibility.
        self.hidden_weights = self.feature_map.hidden_weights
        self.hidden_bias = self.feature_map.hidden_bias
        # One extra output-bias coordinate.
        width = self.hidden_size + 1
        self.weights = np.zeros((self.output_size, width), dtype=np.float64)
        self.inverse_correlation = (
            np.eye(width, dtype=np.float64) / self.regularization
        )
        self.sample_count = np.zeros(1, dtype=np.float64)

    def _features(self, inputs: FloatArray) -> FloatArray:
        return self.feature_map.transform(inputs)

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        return self.weights @ self._features(features)

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        expanded = self._features(features)
        current = self.weights @ expanded
        projected = self.inverse_correlation @ expanded
        gain = projected / (1.0 + float(expanded @ projected))
        self.weights += np.outer(target - current, gain)
        self.inverse_correlation -= np.outer(
            gain, expanded @ self.inverse_correlation
        )
        self.inverse_correlation = 0.5 * (
            self.inverse_correlation + self.inverse_correlation.T
        )
        self.sample_count[0] += 1.0

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.hidden_weights,
            self.hidden_bias,
            self.weights,
            self.inverse_correlation,
            self.sample_count,
        )

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "algorithm": "online_sequential_elm",
            "hidden_size": self.hidden_size,
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "stored_raw_images": 0,
            "stored_feature_vectors": 0,
        }

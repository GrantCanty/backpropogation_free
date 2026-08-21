"""Direct-link Resource-Allocating Network baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines._validation import positive, vector
from continual_core.protocols import FloatArray


@dataclass
class ResourceAllocatingNetworkReadout:
    """Direct-link Resource-Allocating Network with fixed recruited centers."""

    input_size: int
    output_size: int
    max_neurons: int
    learning_rate: float = 0.1
    error_threshold: float = 0.75
    distance_threshold: float = 0.1
    overlap: float = 0.75
    initial_width: float = 0.1
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0 or self.max_neurons <= 0:
            raise ValueError("dimensions and max_neurons must be positive")
        for name in (
            "learning_rate",
            "error_threshold",
            "distance_threshold",
            "overlap",
            "initial_width",
            "epsilon",
        ):
            positive(name, float(getattr(self, name)))
        self.direct_weights = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.neuron_weights = np.zeros(
            (self.output_size, self.max_neurons), dtype=np.float64
        )
        # Compatibility: all trainable output weights in one view is not
        # contiguous, so expose the direct weights for generic introspection.
        self.weights = self.direct_weights
        self.centers = np.zeros(
            (self.max_neurons, self.input_size), dtype=np.float64
        )
        self.widths = np.full(
            self.max_neurons, self.initial_width, dtype=np.float64
        )
        self.active_count = np.zeros(1, dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)
        self.recruitments = np.zeros(1, dtype=np.float64)
        self.capacity_rejections = np.zeros(1, dtype=np.float64)

    def _activities(self, features: FloatArray, count: int) -> FloatArray:
        if count == 0:
            return np.empty(0, dtype=np.float64)
        differences = self.centers[:count] - features
        mean_square = np.mean(np.square(differences), axis=1)
        return np.exp(-mean_square / (2.0 * np.square(self.widths[:count])))

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        count = int(self.active_count[0])
        prediction = self.direct_weights @ features
        if count:
            prediction += self.neuron_weights[:, :count] @ self._activities(
                features, count
            )
        return prediction

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        error = target - prediction
        count = int(self.active_count[0])
        activities = self._activities(features, count)
        nearest_distance = np.inf
        if count:
            distances = np.sqrt(
                np.mean(np.square(self.centers[:count] - features), axis=1)
            )
            nearest_distance = float(np.min(distances))
        should_recruit = (
            float(np.linalg.norm(error)) >= self.error_threshold
            and nearest_distance >= self.distance_threshold
        )

        # Normalized LMS retains the direct-link RAN's shared online fit.
        expanded_norm = float(features @ features)
        if count:
            expanded_norm += float(activities @ activities)
        scale = self.learning_rate / (self.epsilon + expanded_norm)
        self.direct_weights += scale * np.outer(error, features)
        if count:
            self.neuron_weights[:, :count] += scale * np.outer(
                error, activities
            )

        if should_recruit:
            if count < self.max_neurons:
                self.centers[count] = features
                width = (
                    self.initial_width
                    if not np.isfinite(nearest_distance)
                    else max(
                        self.initial_width,
                        self.overlap * nearest_distance,
                    )
                )
                self.widths[count] = width
                # The new unit has activity one on the recruiting event.
                self.neuron_weights[:, count] = error
                self.active_count[0] += 1.0
                self.recruitments[0] += 1.0
            else:
                self.capacity_rejections[0] += 1.0
        self.sample_count[0] += 1.0

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.direct_weights,
            self.neuron_weights,
            self.centers,
            self.widths,
            self.active_count,
            self.sample_count,
            self.recruitments,
            self.capacity_rejections,
        )

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, float | int | str | bool]:
        count = int(self.active_count[0])
        return {
            "algorithm": "direct_link_resource_allocating_network",
            "neuron_capacity": self.max_neurons,
            "active_neurons": count,
            "capacity_full": count == self.max_neurons,
            "recruitments": int(self.recruitments[0]),
            "capacity_rejections": int(self.capacity_rejections[0]),
            "samples_seen": int(self.sample_count[0]),
            "centers_are_fixed_after_recruitment": True,
            "stored_raw_images": 0,
            "stored_feature_vectors": count,
        }


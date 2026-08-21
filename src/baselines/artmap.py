"""Bounded fast-learning Fuzzy ARTMAP baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines._validation import positive, vector
from continual_core.protocols import FloatArray


@dataclass
class FuzzyARTMAPReadout:
    """Bounded fast-learning Fuzzy ARTMAP for supervised online classification."""

    input_size: int
    output_size: int
    max_categories: int
    vigilance: float = 0.8
    choice: float = 0.001
    learning_rate: float = 1.0
    match_tracking_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0 or self.max_categories <= 0:
            raise ValueError("dimensions and max_categories must be positive")
        if not 0.0 <= self.vigilance <= 1.0:
            raise ValueError("vigilance must be in [0, 1]")
        positive("choice", self.choice)
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        positive("match_tracking_epsilon", self.match_tracking_epsilon)
        width = 2 * self.input_size
        self.category_weights = np.ones(
            (self.max_categories, width), dtype=np.float64
        )
        self.category_labels = np.full(
            self.max_categories, -1.0, dtype=np.float64
        )
        self.active_count = np.zeros(1, dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)
        self.category_creations = np.zeros(1, dtype=np.float64)
        self.match_tracking_resets = np.zeros(1, dtype=np.float64)
        self.capacity_rejections = np.zeros(1, dtype=np.float64)
        # Compatibility name; categories, rather than a linear matrix, are the
        # learned weights.
        self.weights = self.category_weights

    def _complement_code(self, features: FloatArray) -> FloatArray:
        # The signed convolution lies in [-1, 1].  This fixed transform avoids
        # stream-dependent min/max statistics.
        normalized = np.clip(0.5 * (features + 1.0), 0.0, 1.0)
        return np.concatenate((normalized, 1.0 - normalized))

    def _choice_and_match(
        self, coded: FloatArray, count: int
    ) -> tuple[FloatArray, FloatArray]:
        intersections = np.minimum(self.category_weights[:count], coded)
        intersection_norm = np.sum(intersections, axis=1)
        choices = intersection_norm / (
            self.choice + np.sum(self.category_weights[:count], axis=1)
        )
        matches = intersection_norm / max(float(np.sum(coded)), np.finfo(float).tiny)
        return choices, matches

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        count = int(self.active_count[0])
        scores = np.zeros(self.output_size, dtype=np.float64)
        if count == 0:
            return scores
        coded = self._complement_code(features)
        choices, matches = self._choice_and_match(coded, count)
        eligible = np.flatnonzero(matches + np.finfo(float).eps >= self.vigilance)
        if len(eligible) == 0:
            return scores
        for index in eligible:
            label = int(self.category_labels[index])
            scores[label] = max(scores[label], float(choices[index]))
        return scores

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        target_class = int(np.argmax(target))
        coded = self._complement_code(features)
        count = int(self.active_count[0])
        rho = self.vigilance
        selected: int | None = None
        if count:
            choices, matches = self._choice_and_match(coded, count)
            order = np.argsort(-choices, kind="stable")
            for index_value in order:
                index = int(index_value)
                match = float(matches[index])
                if match + np.finfo(float).eps < rho:
                    continue
                if int(self.category_labels[index]) == target_class:
                    selected = index
                    break
                rho = min(1.0 + np.finfo(float).eps, match + self.match_tracking_epsilon)
                self.match_tracking_resets[0] += 1.0

        if selected is not None:
            old = self.category_weights[selected]
            learned = np.minimum(coded, old)
            old *= 1.0 - self.learning_rate
            old += self.learning_rate * learned
        elif count < self.max_categories:
            self.category_weights[count] = coded
            self.category_labels[count] = float(target_class)
            self.active_count[0] += 1.0
            self.category_creations[0] += 1.0
        else:
            self.capacity_rejections[0] += 1.0
        self.sample_count[0] += 1.0

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.category_weights,
            self.category_labels,
            self.active_count,
            self.sample_count,
            self.category_creations,
            self.match_tracking_resets,
            self.capacity_rejections,
        )

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, float | int | str | bool]:
        count = int(self.active_count[0])
        return {
            "algorithm": "bounded_fast_learning_fuzzy_artmap",
            "category_capacity": self.max_categories,
            "active_categories": count,
            "capacity_full": count == self.max_categories,
            "category_creations": int(self.category_creations[0]),
            "match_tracking_resets": int(self.match_tracking_resets[0]),
            "capacity_rejections": int(self.capacity_rejections[0]),
            "samples_seen": int(self.sample_count[0]),
            "stored_raw_images": 0,
            "stored_feature_vectors": count,
        }

"""Established forward-only online learners used as external baselines.

These implementations intentionally live outside :mod:`no_backprop`.  They
share its small readout protocol, but are literature baselines rather than
components of the proposed learner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from no_backprop.protocol import FloatArray


def _vector(name: str, value: FloatArray, size: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


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
        _positive("input_scale", self.input_scale)
        _positive("regularization", self.regularization)
        rng = np.random.default_rng(self.seed)
        self.hidden_weights = rng.normal(
            0.0,
            self.input_scale / np.sqrt(self.input_size),
            size=(self.hidden_size, self.input_size),
        )
        self.hidden_bias = rng.uniform(
            -self.input_scale, self.input_scale, self.hidden_size
        )
        # One extra output-bias coordinate.
        width = self.hidden_size + 1
        self.weights = np.zeros((self.output_size, width), dtype=np.float64)
        self.inverse_correlation = (
            np.eye(width, dtype=np.float64) / self.regularization
        )
        self.sample_count = np.zeros(1, dtype=np.float64)

    def _features(self, inputs: FloatArray) -> FloatArray:
        hidden = np.tanh(self.hidden_weights @ inputs + self.hidden_bias)
        return np.concatenate((hidden, np.ones(1, dtype=np.float64)))

    def predict(self, features: FloatArray) -> FloatArray:
        features = _vector("features", features, self.input_size)
        return self.weights @ self._features(features)

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = _vector("features", features, self.input_size)
        target = _vector("target", target, self.output_size)
        _vector("prediction", prediction, self.output_size)
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


@dataclass
class ALDKernelRLSReadout:
    """Multi-output ALD-KRLS with a hard, preallocated dictionary budget.

    This follows the full/reduced recursion of Engel, Mannor, and Meir (2004).
    When the approximate-linear-dependency test admits a point and capacity is
    available, the dictionary expands.  Otherwise the datum still performs the
    reduced KRLS coefficient update.  Once full, mature dictionary elements are
    retained rather than silently applying a deletion policy.
    """

    input_size: int
    output_size: int
    max_dictionary_size: int
    kernel_width: float = 0.1
    ald_tolerance: float = 0.01
    jitter: float = 1e-10

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if self.max_dictionary_size <= 0:
            raise ValueError("max_dictionary_size must be positive")
        _positive("kernel_width", self.kernel_width)
        _positive("ald_tolerance", self.ald_tolerance)
        _positive("jitter", self.jitter)
        size = self.max_dictionary_size
        self.dictionary = np.zeros((size, self.input_size), dtype=np.float64)
        self.kernel_inverse = np.zeros((size, size), dtype=np.float64)
        self.projection_inverse = np.zeros((size, size), dtype=np.float64)
        self.coefficients = np.zeros((size, self.output_size), dtype=np.float64)
        # Compatibility name for generic learner introspection.
        self.weights = self.coefficients.T
        self.active_count = np.zeros(1, dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)
        self.dictionary_admissions = np.zeros(1, dtype=np.float64)
        self.ald_rejections = np.zeros(1, dtype=np.float64)
        self.capacity_rejections = np.zeros(1, dtype=np.float64)

    def _kernel_vector(self, features: FloatArray, count: int) -> FloatArray:
        if count == 0:
            return np.empty(0, dtype=np.float64)
        differences = self.dictionary[:count] - features
        mean_square = np.mean(np.square(differences), axis=1)
        return np.exp(-mean_square / (2.0 * self.kernel_width**2))

    def predict(self, features: FloatArray) -> FloatArray:
        features = _vector("features", features, self.input_size)
        count = int(self.active_count[0])
        if count == 0:
            return np.zeros(self.output_size, dtype=np.float64)
        kernels = self._kernel_vector(features, count)
        return kernels @ self.coefficients[:count]

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = _vector("features", features, self.input_size)
        target = _vector("target", target, self.output_size)
        prediction = _vector("prediction", prediction, self.output_size)
        count = int(self.active_count[0])
        if count == 0:
            self.dictionary[0] = features
            self.kernel_inverse[0, 0] = 1.0
            self.projection_inverse[0, 0] = 1.0
            self.coefficients[0] = target
            self.active_count[0] = 1.0
            self.dictionary_admissions[0] += 1.0
            self.sample_count[0] += 1.0
            return

        kernels = self._kernel_vector(features, count)
        kernel_inverse = self.kernel_inverse[:count, :count]
        projection_inverse = self.projection_inverse[:count, :count]
        projection = kernel_inverse @ kernels
        delta = max(0.0, 1.0 - float(kernels @ projection))
        error = target - prediction
        can_expand = (
            delta > self.ald_tolerance and count < self.max_dictionary_size
        )
        if can_expand:
            safe_delta = max(delta, self.jitter)
            old_kernel_inverse = kernel_inverse.copy()
            self.kernel_inverse[:count, :count] = (
                old_kernel_inverse + np.outer(projection, projection) / safe_delta
            )
            self.kernel_inverse[:count, count] = -projection / safe_delta
            self.kernel_inverse[count, :count] = -projection / safe_delta
            self.kernel_inverse[count, count] = 1.0 / safe_delta
            self.projection_inverse[count, :count].fill(0.0)
            self.projection_inverse[:count, count].fill(0.0)
            self.projection_inverse[count, count] = 1.0
            self.coefficients[:count] -= np.outer(
                projection / safe_delta, error
            )
            self.coefficients[count] = error / safe_delta
            self.dictionary[count] = features
            self.active_count[0] += 1.0
            self.dictionary_admissions[0] += 1.0
        else:
            projected = projection_inverse @ projection
            denominator = 1.0 + float(projection @ projected)
            gain = projected / max(denominator, self.jitter)
            self.projection_inverse[:count, :count] -= np.outer(
                gain, projection @ projection_inverse
            )
            coefficient_gain = kernel_inverse @ gain
            self.coefficients[:count] += np.outer(coefficient_gain, error)
            if delta > self.ald_tolerance:
                self.capacity_rejections[0] += 1.0
            else:
                self.ald_rejections[0] += 1.0
        self.sample_count[0] += 1.0

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.dictionary,
            self.kernel_inverse,
            self.projection_inverse,
            self.coefficients,
            self.active_count,
            self.sample_count,
            self.dictionary_admissions,
            self.ald_rejections,
            self.capacity_rejections,
        )

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, float | int | str | bool]:
        count = int(self.active_count[0])
        return {
            "algorithm": "ald_kernel_recursive_least_squares",
            "dictionary_capacity": self.max_dictionary_size,
            "active_dictionary_vectors": count,
            "capacity_full": count == self.max_dictionary_size,
            "dictionary_admissions": int(self.dictionary_admissions[0]),
            "ald_reduced_updates": int(self.ald_rejections[0]),
            "capacity_reduced_updates": int(self.capacity_rejections[0]),
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "stored_raw_images": 0,
            "stored_feature_vectors": count,
            "dictionary_elements_are_observed_features": True,
        }


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
            _positive(name, float(getattr(self, name)))
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
        features = _vector("features", features, self.input_size)
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
        features = _vector("features", features, self.input_size)
        target = _vector("target", target, self.output_size)
        prediction = _vector("prediction", prediction, self.output_size)
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
        _positive("choice", self.choice)
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        _positive("match_tracking_epsilon", self.match_tracking_epsilon)
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
        features = _vector("features", features, self.input_size)
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
        features = _vector("features", features, self.input_size)
        target = _vector("target", target, self.output_size)
        _vector("prediction", prediction, self.output_size)
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

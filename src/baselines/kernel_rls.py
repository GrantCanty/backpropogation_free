"""Budgeted approximate-linear-dependency kernel RLS baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines._validation import positive, vector
from continual_core.protocols import FloatArray


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
        positive("kernel_width", self.kernel_width)
        positive("ald_tolerance", self.ald_tolerance)
        positive("jitter", self.jitter)
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

    def _kernelvector(self, features: FloatArray, count: int) -> FloatArray:
        if count == 0:
            return np.empty(0, dtype=np.float64)
        differences = self.dictionary[:count] - features
        mean_square = np.mean(np.square(differences), axis=1)
        return np.exp(-mean_square / (2.0 * self.kernel_width**2))

    def predict(self, features: FloatArray) -> FloatArray:
        features = vector("features", features, self.input_size)
        count = int(self.active_count[0])
        if count == 0:
            return np.zeros(self.output_size, dtype=np.float64)
        kernels = self._kernelvector(features, count)
        return kernels @ self.coefficients[:count]

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
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

        kernels = self._kernelvector(features, count)
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


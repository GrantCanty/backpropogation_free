"""Closed-form and local online readout update rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from no_backprop.protocol import FloatArray


class Readout(Protocol):
    input_size: int
    output_size: int

    def predict(self, features: FloatArray) -> FloatArray: ...

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None: ...

    @property
    def state_nbytes(self) -> int: ...


def _validate_vector(name: str, value: FloatArray, size: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape {(size,)}, got {array.shape}")
    return array


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
        features = _validate_vector("features", features, self.input_size)
        return self.weights @ features

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        _validate_vector("features", features, self.input_size)
        _validate_vector("target", target, self.output_size)
        _validate_vector("prediction", prediction, self.output_size)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes


@dataclass
class LMSReadout(FrozenReadout):
    learning_rate: float = 0.2
    normalized: bool = True
    epsilon: float = 1e-6
    update_clip: float | None = 1.0

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        error = target - prediction
        scale = self.learning_rate
        if self.normalized:
            scale /= self.epsilon + float(features @ features)
        update = scale * np.outer(error, features)
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.weights += update


@dataclass
class RLSReadout(FrozenReadout):
    regularization: float = 1.0
    forgetting_factor: float = 0.999

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        self.inverse_correlation = (
            np.eye(self.input_size, dtype=np.float64) / self.regularization
        )

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        projected = self.inverse_correlation @ features
        denominator = self.forgetting_factor + float(features @ projected)
        gain = projected / denominator
        error = target - prediction
        self.weights += np.outer(error, gain)
        feature_times_inverse = features @ self.inverse_correlation
        self.inverse_correlation = (
            self.inverse_correlation - np.outer(gain, feature_times_inverse)
        ) / self.forgetting_factor
        self.inverse_correlation = 0.5 * (
            self.inverse_correlation + self.inverse_correlation.T
        )

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes + self.inverse_correlation.nbytes


@dataclass
class DiagonalRLSReadout(FrozenReadout):
    """Linear-memory approximation that retains only RLS's diagonal state."""

    regularization: float = 1.0
    forgetting_factor: float = 0.999

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        self.inverse_diagonal = np.full(
            self.input_size, 1.0 / self.regularization, dtype=np.float64
        )

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        projected = self.inverse_diagonal * features
        denominator = self.forgetting_factor + float(features @ projected)
        gain = projected / denominator
        self.weights += np.outer(target - prediction, gain)
        self.inverse_diagonal = (
            self.inverse_diagonal - np.square(projected) / denominator
        ) / self.forgetting_factor
        np.maximum(self.inverse_diagonal, np.finfo(float).tiny, out=self.inverse_diagonal)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes + self.inverse_diagonal.nbytes


@dataclass
class BlockRLSReadout(FrozenReadout):
    """RLS approximation that preserves correlations within feature blocks."""

    regularization: float = 1.0
    forgetting_factor: float = 0.999
    block_size: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if not 0.0 < self.forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_slices: list[slice] = []
        self.inverse_blocks: list[FloatArray] = []
        for start in range(0, self.input_size, self.block_size):
            stop = min(start + self.block_size, self.input_size)
            self.block_slices.append(slice(start, stop))
            self.inverse_blocks.append(
                np.eye(stop - start, dtype=np.float64) / self.regularization
            )

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        projected_blocks = [
            inverse @ features[block]
            for block, inverse in zip(self.block_slices, self.inverse_blocks)
        ]
        denominator = self.forgetting_factor + sum(
            float(features[block] @ projected)
            for block, projected in zip(self.block_slices, projected_blocks)
        )
        gain = np.concatenate(projected_blocks) / denominator
        self.weights += np.outer(target - prediction, gain)
        for index, (inverse, projected) in enumerate(
            zip(self.inverse_blocks, projected_blocks)
        ):
            updated = (
                inverse - np.outer(projected, projected) / denominator
            ) / self.forgetting_factor
            self.inverse_blocks[index] = 0.5 * (updated + updated.T)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes + sum(block.nbytes for block in self.inverse_blocks)


@dataclass
class PrototypeReadout:
    """Protected per-class semantic memory with linear update cost."""

    input_size: int
    output_size: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.centroids = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.counts = np.zeros(self.output_size, dtype=np.float64)

    @property
    def weights(self) -> FloatArray:
        return self.centroids

    def predict(self, features: FloatArray) -> FloatArray:
        features = _validate_vector("features", features, self.input_size)
        if not np.any(self.counts > 0.0):
            return np.zeros(self.output_size, dtype=np.float64)
        distances = np.mean(np.square(self.centroids - features), axis=1)
        scores = -distances
        seen = self.counts > 0.0
        scores[~seen] = float(np.min(scores[seen]) - 1.0)
        return scores

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        _validate_vector("prediction", prediction, self.output_size)
        class_index = int(np.argmax(target))
        self.counts[class_index] += 1.0
        rate = 1.0 / self.counts[class_index]
        self.centroids[class_index] += rate * (
            features - self.centroids[class_index]
        )

    @property
    def state_nbytes(self) -> int:
        return self.centroids.nbytes + self.counts.nbytes


def _prediction_margin(scores: FloatArray) -> float:
    """Return a scale-bounded gap between the largest two scores."""

    if len(scores) < 2:
        return 1.0
    largest = np.partition(scores, -2)[-2:]
    gap = float(largest[-1] - largest[-2])
    return gap / (1.0 + abs(float(largest[-1])) + abs(float(largest[-2])))


@dataclass
class CumulativeMemoryReadout:
    """Factor-free complementary memory with cumulative sufficient statistics.

    The slow path is exact online ridge regression with unit weight for every
    observation.  The fast path compresses recurring slow-path errors into one
    prototype per (target class, mistaken class) pair.  A cumulative reliability
    ranker observes both candidates and learns which one is more reliable.

    No state is aged, decayed, replayed, or indexed by a history window.  The
    model stores sufficient statistics and error representations rather than
    raw observations.
    """

    input_size: int
    output_size: int
    seed: int = 0
    regularization: float = 1.0
    rank_bins: int = 16

    def __post_init__(self) -> None:
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.rank_bins <= 0:
            raise ValueError("rank_bins must be positive")
        self.slow_weights = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.slow_inverse_correlation = (
            np.eye(self.input_size, dtype=np.float64) / self.regularization
        )
        self.semantic_centroids = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.semantic_counts = np.zeros(self.output_size, dtype=np.float64)

        # A cell represents a recurring way in which target class i was
        # mistaken for class j.  Counts make each centroid the exact cumulative
        # mean of all errors assigned to that semantic cell.
        self.exception_centroids = np.zeros(
            (self.output_size, self.output_size, self.input_size),
            dtype=np.float64,
        )
        self.exception_counts = np.zeros(
            (self.output_size, self.output_size), dtype=np.float64
        )

        # The ranker records counterfactual correctness for both systems in a
        # compact context table: slow class, fast class, slow-margin bin, and
        # relative-proximity bin.  It defaults to the slow path until
        # cumulative evidence favors fast memory in the same context.
        rank_shape = (
            self.output_size,
            self.output_size,
            self.rank_bins,
            self.rank_bins,
        )
        self.rank_trials = np.zeros(rank_shape, dtype=np.float64)
        self.rank_correct = np.zeros((2, *rank_shape), dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)
        self.rank_update_count = np.zeros(1, dtype=np.float64)
        self.selection_counts = np.zeros(2, dtype=np.float64)

        self._last_slow_prediction: FloatArray | None = None
        self._last_fast_prediction: FloatArray | None = None
        self._last_rank_cell: tuple[int, int, int, int] | None = None
        self._last_fast_available = False
        self._last_selection = 0

    @property
    def weights(self) -> FloatArray:
        """Expose the semantic weights for generic readout inspection."""

        return self.slow_weights

    def _fast_candidate(
        self, features: FloatArray
    ) -> tuple[FloatArray, bool, float, float]:
        active = self.exception_counts > 0.0
        if not np.any(active):
            return (
                np.zeros(self.output_size, dtype=np.float64),
                False,
                0.0,
                float("inf"),
            )

        class_distances = np.full(self.output_size, np.inf, dtype=np.float64)
        for class_index in range(self.output_size):
            class_active = active[class_index]
            if not np.any(class_active):
                continue
            differences = (
                self.exception_centroids[class_index, class_active] - features
            )
            class_distances[class_index] = float(
                np.min(np.mean(np.square(differences), axis=1))
            )

        finite = np.isfinite(class_distances)
        fast_class = int(np.argmin(class_distances))
        prediction = np.zeros(self.output_size, dtype=np.float64)
        prediction[fast_class] = 1.0
        best_distance = float(class_distances[fast_class])
        finite_distances = np.sort(class_distances[finite])
        if len(finite_distances) < 2:
            margin = 1.0 / (1.0 + best_distance)
        else:
            margin = float(
                (finite_distances[1] - finite_distances[0])
                / (1.0 + finite_distances[1] + finite_distances[0])
            )
        return prediction, True, margin, best_distance

    def predict(self, features: FloatArray) -> FloatArray:
        features = _validate_vector("features", features, self.input_size)
        slow_prediction = self.slow_weights @ features
        fast_prediction, fast_available, _fast_margin, fast_distance = (
            self._fast_candidate(features)
        )
        slow_class = int(np.argmax(slow_prediction))
        fast_class = int(np.argmax(fast_prediction)) if fast_available else slow_class
        if self.semantic_counts[slow_class] > 0.0:
            slow_distance = float(
                np.mean(
                    np.square(self.semantic_centroids[slow_class] - features)
                )
            )
        else:
            slow_distance = float("inf")
        if np.isfinite(slow_distance) and np.isfinite(fast_distance):
            relative_fast_proximity = slow_distance / (
                np.finfo(float).eps + slow_distance + fast_distance
            )
        elif np.isfinite(fast_distance):
            relative_fast_proximity = 1.0
        else:
            relative_fast_proximity = 0.0
        slow_margin = _prediction_margin(slow_prediction)
        slow_margin_bin = min(
            int(np.clip(slow_margin, 0.0, 1.0) * self.rank_bins),
            self.rank_bins - 1,
        )
        proximity_bin = min(
            int(np.clip(relative_fast_proximity, 0.0, 1.0) * self.rank_bins),
            self.rank_bins - 1,
        )
        rank_cell = (slow_class, fast_class, slow_margin_bin, proximity_bin)
        disagreement = fast_available and slow_class != fast_class
        trials = self.rank_trials[rank_cell]
        selection = 0
        if disagreement and relative_fast_proximity > 0.5 and trials > 0.0:
            selection = int(
                self.rank_correct[(1, *rank_cell)]
                > self.rank_correct[(0, *rank_cell)]
            )

        self._last_slow_prediction = slow_prediction.copy()
        self._last_fast_prediction = fast_prediction.copy()
        self._last_rank_cell = rank_cell
        self._last_fast_available = fast_available
        self._last_selection = selection
        return (fast_prediction if selection else slow_prediction).copy()

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        _validate_vector("prediction", prediction, self.output_size)
        if self._last_slow_prediction is None or self._last_rank_cell is None:
            raise RuntimeError("predict must be called before update")

        target_class = int(np.argmax(target))
        slow_class = int(np.argmax(self._last_slow_prediction))

        # Train the reliability ranker on both observable counterfactuals.
        # Counts give every disagreement unit weight and never decay.
        fast_class = (
            int(np.argmax(self._last_fast_prediction))
            if self._last_fast_prediction is not None
            else slow_class
        )
        if self._last_fast_available and fast_class != slow_class:
            rank_cell = self._last_rank_cell
            self.rank_trials[rank_cell] += 1.0
            self.rank_correct[(0, *rank_cell)] += float(slow_class == target_class)
            self.rank_correct[(1, *rank_cell)] += float(fast_class == target_class)
            self.rank_update_count[0] += 1.0

        # The fast representation sees only the semantic system's residual
        # cases, enforcing complementary rather than duplicate learning.
        if slow_class != target_class:
            count = self.exception_counts[target_class, slow_class] + 1.0
            centroid = self.exception_centroids[target_class, slow_class]
            centroid += (features - centroid) / count
            self.exception_counts[target_class, slow_class] = count

        semantic_count = self.semantic_counts[target_class] + 1.0
        semantic_centroid = self.semantic_centroids[target_class]
        semantic_centroid += (features - semantic_centroid) / semantic_count
        self.semantic_counts[target_class] = semantic_count

        # Exact recursive ridge update with an implicit observation weight of
        # one.  Every sample remains in the cumulative least-squares objective.
        projected = self.slow_inverse_correlation @ features
        denominator = 1.0 + float(features @ projected)
        gain = projected / denominator
        self.slow_weights += np.outer(
            target - self._last_slow_prediction, gain
        )
        self.slow_inverse_correlation -= np.outer(
            gain, features @ self.slow_inverse_correlation
        )
        self.slow_inverse_correlation = 0.5 * (
            self.slow_inverse_correlation + self.slow_inverse_correlation.T
        )

        self.sample_count[0] += 1.0
        self.selection_counts[self._last_selection] += 1.0
        self._last_slow_prediction = None
        self._last_fast_prediction = None
        self._last_rank_cell = None

    @property
    def diagnostics(self) -> dict[str, float | int]:
        total_selections = float(np.sum(self.selection_counts))
        return {
            "samples_in_slow_statistics": int(self.sample_count[0]),
            "stored_raw_samples": 0,
            "active_exception_representations": int(
                np.count_nonzero(self.exception_counts)
            ),
            "active_semantic_representations": int(
                np.count_nonzero(self.semantic_counts)
            ),
            "errors_compressed_into_fast_memory": int(
                np.sum(self.exception_counts)
            ),
            "rank_updates": int(self.rank_update_count[0]),
            "slow_selection_rate": (
                0.0
                if total_selections == 0.0
                else float(self.selection_counts[0] / total_selections)
            ),
            "fast_selection_rate": (
                0.0
                if total_selections == 0.0
                else float(self.selection_counts[1] / total_selections)
            ),
        }

    @property
    def state_nbytes(self) -> int:
        arrays = (
            self.slow_weights,
            self.slow_inverse_correlation,
            self.semantic_centroids,
            self.semantic_counts,
            self.exception_centroids,
            self.exception_counts,
            self.rank_trials,
            self.rank_correct,
            self.sample_count,
            self.rank_update_count,
            self.selection_counts,
        )
        return sum(array.nbytes for array in arrays)


def _normalized_entropy(scores: FloatArray) -> float:
    """Return categorical entropy in [0, 1] from finite uncalibrated scores."""

    if len(scores) <= 1:
        return 0.0
    shifted = scores - float(np.max(scores))
    exponentials = np.exp(np.clip(shifted, -700.0, 0.0))
    probabilities = exponentials / np.sum(exponentials)
    entropy = -float(
        np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)))
    )
    return entropy / float(np.log(len(scores)))


@dataclass
class CumulativeMaturityReadout:
    """Single-path expanding representation with evidence-based maturation.

    Every observation updates one cumulative ridge model.  Recruitable radial
    neurons extend the shared representation without producing a second expert
    prediction or requiring a router.  A neuron's cumulative activation energy
    is its maturity evidence; neither model variant ages or decays state.

    When ``entropy_gated`` is true, an error recruits capacity only when its
    predictive entropy is below the cumulative mean entropy of correct
    predictions.  When ``leverage_gated`` is true, recruitment requires the
    current normalized RLS leverage to be below its cumulative pre-update mean:
    an error must occur in a region that the model has already observed enough
    to consider familiar.  The matched control recruits on any
    representationally distinct error.
    """

    input_size: int
    output_size: int
    seed: int = 0
    regularization: float = 1.0
    max_neurons: int = 32
    rbf_width: float = 0.05
    min_center_distance: float = 0.01
    entropy_gated: bool = False
    leverage_gated: bool = False

    def __post_init__(self) -> None:
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.max_neurons < 0:
            raise ValueError("max_neurons cannot be negative")
        if self.rbf_width <= 0.0:
            raise ValueError("rbf_width must be positive")
        if self.min_center_distance < 0.0:
            raise ValueError("min_center_distance cannot be negative")
        if self.entropy_gated and self.leverage_gated:
            raise ValueError("entropy and leverage gates are mutually exclusive")

        self.expanded_size = self.input_size + self.max_neurons
        self.expanded_weights = np.zeros(
            (self.output_size, self.expanded_size), dtype=np.float64
        )
        self.inverse_correlation = (
            np.eye(self.expanded_size, dtype=np.float64) / self.regularization
        )
        self.neuron_centers = np.zeros(
            (self.max_neurons, self.input_size), dtype=np.float64
        )
        self.neuron_active = np.zeros(self.max_neurons, dtype=np.float64)
        self.neuron_evidence = np.zeros(self.max_neurons, dtype=np.float64)
        self.neuron_labels = np.full(self.max_neurons, -1.0, dtype=np.float64)
        self.neuron_recruitment_entropy = np.zeros(
            self.max_neurons, dtype=np.float64
        )
        self.active_count = np.zeros(1, dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)
        self.correct_entropy_sum = np.zeros(1, dtype=np.float64)
        self.correct_entropy_count = np.zeros(1, dtype=np.float64)
        self.error_count = np.zeros(1, dtype=np.float64)
        self.recruitment_candidate_count = np.zeros(1, dtype=np.float64)
        self.entropy_rejection_count = np.zeros(1, dtype=np.float64)
        self.leverage_rejection_count = np.zeros(1, dtype=np.float64)
        self.proximity_rejection_count = np.zeros(1, dtype=np.float64)
        self.normalized_leverage_sum = np.zeros(1, dtype=np.float64)
        self.normalized_leverage_count = np.zeros(1, dtype=np.float64)

        self._last_features: FloatArray | None = None
        self._last_expanded: FloatArray | None = None
        self._last_prediction: FloatArray | None = None
        self._last_entropy = 1.0
        self._last_normalized_leverage = 1.0

    @property
    def weights(self) -> FloatArray:
        return self.expanded_weights

    def _activities(self, features: FloatArray) -> FloatArray:
        activities = np.zeros(self.max_neurons, dtype=np.float64)
        count = int(self.active_count[0])
        if count == 0:
            return activities
        differences = self.neuron_centers[:count] - features
        mean_square_distance = np.mean(np.square(differences), axis=1)
        activities[:count] = np.exp(
            -mean_square_distance / (2.0 * self.rbf_width**2)
        )
        return activities

    def _expanded_features(self, features: FloatArray) -> FloatArray:
        return np.concatenate((features, self._activities(features)))

    def predict(self, features: FloatArray) -> FloatArray:
        features = _validate_vector("features", features, self.input_size)
        expanded = self._expanded_features(features)
        prediction = self.expanded_weights @ expanded
        self._last_features = features.copy()
        self._last_expanded = expanded
        self._last_prediction = prediction.copy()
        self._last_entropy = _normalized_entropy(prediction)
        return prediction.copy()

    def _is_distinct(self, features: FloatArray) -> bool:
        count = int(self.active_count[0])
        if count == 0:
            return True
        distances = np.mean(
            np.square(self.neuron_centers[:count] - features), axis=1
        )
        return float(np.min(distances)) >= self.min_center_distance

    def _entropy_allows_recruitment(self) -> bool:
        if not self.entropy_gated:
            return True
        correct_count = self.correct_entropy_count[0]
        if correct_count == 0.0:
            return False
        correct_mean = self.correct_entropy_sum[0] / correct_count
        return self._last_entropy + np.finfo(float).eps < correct_mean

    def _leverage_allows_recruitment(self) -> bool:
        if not self.leverage_gated:
            return True
        count = self.normalized_leverage_count[0]
        if count == 0.0:
            return False
        mean = self.normalized_leverage_sum[0] / count
        return self._last_normalized_leverage + np.finfo(float).eps < mean

    def _gate_allows_recruitment(self) -> bool:
        return (
            self._entropy_allows_recruitment()
            and self._leverage_allows_recruitment()
        )

    def _recruit(self, features: FloatArray, target_class: int) -> bool:
        index = int(self.active_count[0])
        self.neuron_centers[index] = features
        self.neuron_active[index] = 1.0
        self.neuron_labels[index] = float(target_class)
        self.neuron_recruitment_entropy[index] = self._last_entropy
        self.active_count[0] += 1.0
        return True

    def _prepare_representation_update(
        self, features: FloatArray, target_class: int
    ) -> bool:
        """Allow a subclass to activate prepared features before learning."""

        return False

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        _validate_vector("prediction", prediction, self.output_size)
        if (
            self._last_features is None
            or self._last_expanded is None
            or self._last_prediction is None
        ):
            raise RuntimeError("predict must be called before update")

        projected = self.inverse_correlation @ self._last_expanded
        leverage = max(0.0, float(self._last_expanded @ projected))
        self._last_normalized_leverage = leverage / (1.0 + leverage)
        target_class = int(np.argmax(target))
        prepared = self._prepare_representation_update(features, target_class)
        if prepared:
            self._last_expanded = self._expanded_features(features)
            projected = self.inverse_correlation @ self._last_expanded
        active_before_update = int(self.active_count[0])
        predicted_class = int(np.argmax(self._last_prediction))
        correct = target_class == predicted_class
        if correct:
            self.correct_entropy_sum[0] += self._last_entropy
            self.correct_entropy_count[0] += 1.0
        else:
            self.error_count[0] += 1.0
            self.recruitment_candidate_count[0] += 1.0
            capacity_available = int(self.active_count[0]) < self.max_neurons
            gate_allowed = self._gate_allows_recruitment()
            distinct = self._is_distinct(features)
            if capacity_available and gate_allowed and distinct:
                activated = self._recruit(features, target_class)
                if activated:
                    # The promoting observation immediately trains the unit.
                    self._last_expanded = self._expanded_features(features)
                    projected = self.inverse_correlation @ self._last_expanded
            elif capacity_available and not gate_allowed and self.entropy_gated:
                self.entropy_rejection_count[0] += 1.0
            elif capacity_available and not gate_allowed and self.leverage_gated:
                self.leverage_rejection_count[0] += 1.0
            elif capacity_available and not distinct:
                self.proximity_rejection_count[0] += 1.0

        expanded = self._last_expanded
        prediction_for_update = self.expanded_weights @ expanded
        denominator = 1.0 + float(expanded @ projected)
        gain = projected / denominator
        self.expanded_weights += np.outer(target - prediction_for_update, gain)
        self.inverse_correlation -= np.outer(
            gain, expanded @ self.inverse_correlation
        )
        self.inverse_correlation = 0.5 * (
            self.inverse_correlation + self.inverse_correlation.T
        )
        self._update_neuron_statistics(
            features, expanded[self.input_size :], active_before_update
        )
        self.normalized_leverage_sum[0] += self._last_normalized_leverage
        self.normalized_leverage_count[0] += 1.0
        self.sample_count[0] += 1.0

        self._last_features = None
        self._last_expanded = None
        self._last_prediction = None

    def _update_neuron_statistics(
        self,
        features: FloatArray,
        activities: FloatArray,
        active_before_update: int,
    ) -> None:
        """Accumulate evidence; subclasses may also adapt receptive fields."""

        if self.max_neurons:
            self.neuron_evidence += np.square(activities)

    @property
    def diagnostics(self) -> dict[str, float | int | bool]:
        count = int(self.active_count[0])
        evidence = self.neuron_evidence[:count]
        maturity = evidence / (self.regularization + evidence)
        correct_count = self.correct_entropy_count[0]
        leverage_count = self.normalized_leverage_count[0]
        return {
            "entropy_gated": self.entropy_gated,
            "leverage_gated": self.leverage_gated,
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "stored_raw_samples": 0,
            "active_neurons": count,
            "available_neurons": self.max_neurons - count,
            "mean_neuron_maturity": (
                0.0 if count == 0 else float(np.mean(maturity))
            ),
            "minimum_neuron_maturity": (
                0.0 if count == 0 else float(np.min(maturity))
            ),
            "maximum_neuron_maturity": (
                0.0 if count == 0 else float(np.max(maturity))
            ),
            "observed_errors": int(self.error_count[0]),
            "recruitment_candidates": int(self.recruitment_candidate_count[0]),
            "entropy_rejections": int(self.entropy_rejection_count[0]),
            "leverage_rejections": int(self.leverage_rejection_count[0]),
            "proximity_rejections": int(self.proximity_rejection_count[0]),
            "mean_correct_prediction_entropy": (
                0.0
                if correct_count == 0.0
                else float(self.correct_entropy_sum[0] / correct_count)
            ),
            "mean_normalized_leverage": (
                0.0
                if leverage_count == 0.0
                else float(self.normalized_leverage_sum[0] / leverage_count)
            ),
            "samples_in_leverage_statistics": int(leverage_count),
        }

    @property
    def state_nbytes(self) -> int:
        arrays = (
            self.expanded_weights,
            self.inverse_correlation,
            self.neuron_centers,
            self.neuron_active,
            self.neuron_evidence,
            self.neuron_labels,
            self.neuron_recruitment_entropy,
            self.active_count,
            self.sample_count,
            self.correct_entropy_sum,
            self.correct_entropy_count,
            self.error_count,
            self.recruitment_candidate_count,
            self.entropy_rejection_count,
            self.leverage_rejection_count,
            self.proximity_rejection_count,
            self.normalized_leverage_sum,
            self.normalized_leverage_count,
        )
        return sum(array.nbytes for array in arrays)


@dataclass
class ProbationaryMaturityReadout(CumulativeMaturityReadout):
    """Leverage-gated local neurons that learn before becoming immutable.

    The first qualifying error creates a dormant candidate.  A later nearby
    observation with the same target is its confirmation: their cumulative
    centroid becomes an active key and is frozen permanently.  Dormant
    candidates never participate in prediction, so no historical RLS statistic
    is formed while a key is moving.
    """

    def __post_init__(self) -> None:
        self.leverage_gated = True
        super().__post_init__()
        self.candidate_centers = np.zeros(
            (self.max_neurons, self.input_size), dtype=np.float64
        )
        self.candidate_counts = np.zeros(self.max_neurons, dtype=np.float64)
        self.candidate_labels = np.full(
            self.max_neurons, -1.0, dtype=np.float64
        )
        self.candidate_active = np.zeros(self.max_neurons, dtype=np.float64)
        self.candidates_created = np.zeros(1, dtype=np.float64)
        self.candidates_promoted = np.zeros(1, dtype=np.float64)
        self.candidate_pool_rejections = np.zeros(1, dtype=np.float64)

    def _matching_candidate(
        self, features: FloatArray, target_class: int
    ) -> int | None:
        eligible = np.flatnonzero(
            (self.candidate_active > 0.0)
            & (self.candidate_labels == float(target_class))
        )
        if len(eligible) == 0:
            return None
        distances = np.mean(
            np.square(self.candidate_centers[eligible] - features), axis=1
        )
        nearest_position = int(np.argmin(distances))
        if float(distances[nearest_position]) > self.min_center_distance:
            return None
        return int(eligible[nearest_position])

    def _empty_candidate_slot(self) -> int | None:
        available = np.flatnonzero(self.candidate_active == 0.0)
        return None if len(available) == 0 else int(available[0])

    def _clear_candidate(self, index: int) -> None:
        self.candidate_centers[index].fill(0.0)
        self.candidate_counts[index] = 0.0
        self.candidate_labels[index] = -1.0
        self.candidate_active[index] = 0.0

    def _recruit(self, features: FloatArray, target_class: int) -> bool:
        candidate_index = self._empty_candidate_slot()
        if candidate_index is None:
            self.candidate_pool_rejections[0] += 1.0
            return False
        self.candidate_centers[candidate_index] = features
        self.candidate_counts[candidate_index] = 1.0
        self.candidate_labels[candidate_index] = float(target_class)
        self.candidate_active[candidate_index] = 1.0
        self.candidates_created[0] += 1.0
        return False

    def _prepare_representation_update(
        self, features: FloatArray, target_class: int
    ) -> bool:
        if int(self.active_count[0]) >= self.max_neurons:
            return False
        candidate_index = self._matching_candidate(features, target_class)
        if candidate_index is None:
            return False
        old_count = self.candidate_counts[candidate_index]
        new_count = old_count + 1.0
        center = self.candidate_centers[candidate_index]
        center += (features - center) / new_count
        self.candidate_counts[candidate_index] = new_count
        frozen_center = center.copy()
        self._clear_candidate(candidate_index)
        super()._recruit(frozen_center, target_class)
        self.candidates_promoted[0] += 1.0
        return True

    @property
    def diagnostics(self) -> dict[str, float | int | bool]:
        result = super().diagnostics
        pending = self.candidate_counts[self.candidate_active > 0.0]
        result.update(
            {
                "probationary_keys": True,
                "active_keys_are_frozen": True,
                "pending_candidates": int(np.sum(self.candidate_active)),
                "candidates_created": int(self.candidates_created[0]),
                "candidates_promoted": int(self.candidates_promoted[0]),
                "candidate_pool_rejections": int(
                    self.candidate_pool_rejections[0]
                ),
                "mean_pending_candidate_evidence": (
                    0.0 if len(pending) == 0 else float(np.mean(pending))
                ),
            }
        )
        return result

    @property
    def state_nbytes(self) -> int:
        return super().state_nbytes + sum(
            array.nbytes
            for array in (
                self.candidate_centers,
                self.candidate_counts,
                self.candidate_labels,
                self.candidate_active,
                self.candidates_created,
                self.candidates_promoted,
                self.candidate_pool_rejections,
            )
        )


@dataclass
class ResponsibleProbationaryMaturityReadout(ProbationaryMaturityReadout):
    """Probationary memory with dynamically sparse local-key responsibility."""

    responsibility_k: int = 4
    normalize_responsibility: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.responsibility_k <= 0:
            raise ValueError("responsibility_k must be positive")
        self.responsible_key_sum = np.zeros(1, dtype=np.float64)
        self.responsibility_sample_count = np.zeros(1, dtype=np.float64)

    def _activities(self, features: FloatArray) -> FloatArray:
        activities = super()._activities(features)
        count = int(self.active_count[0])
        keep = min(count, self.responsibility_k)
        if keep == 0:
            return activities
        if keep < count:
            active_values = activities[:count]
            selected = np.argpartition(active_values, -keep)[-keep:]
            sparse = np.zeros_like(activities)
            sparse[selected] = active_values[selected]
            activities = sparse
        if self.normalize_responsibility:
            total = float(np.sum(activities))
            if total > np.finfo(float).tiny:
                activities /= total
        return activities

    def _update_neuron_statistics(
        self,
        features: FloatArray,
        activities: FloatArray,
        active_before_update: int,
    ) -> None:
        super()._update_neuron_statistics(
            features, activities, active_before_update
        )
        self.responsible_key_sum[0] += float(np.count_nonzero(activities))
        self.responsibility_sample_count[0] += 1.0

    @property
    def diagnostics(self) -> dict[str, float | int | bool]:
        result = super().diagnostics
        count = self.responsibility_sample_count[0]
        result.update(
            {
                "sparse_local_responsibility": True,
                "responsibility_k": self.responsibility_k,
                "normalized_responsibility": self.normalize_responsibility,
                "mean_responsible_keys": (
                    0.0
                    if count == 0.0
                    else float(self.responsible_key_sum[0] / count)
                ),
            }
        )
        return result

    @property
    def state_nbytes(self) -> int:
        return (
            super().state_nbytes
            + self.responsible_key_sum.nbytes
            + self.responsibility_sample_count.nbytes
        )


@dataclass
class KeyValueMaturityReadout(CumulativeMaturityReadout):
    """Maturity network whose recruited keys and locality learn cumulatively.

    Each local neuron is an unnormalized attention-like key/value pair.  Its
    center is the key, diagonal variance defines key similarity, and its column
    in ``expanded_weights`` is the value.  Soft activation-weighted Welford
    statistics move developing keys and reduce movement as evidence grows.
    """

    key_prior_strength: float = 4.0
    minimum_key_variance: float = 4e-4
    maximum_key_variance: float = 3.6e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.key_prior_strength <= 0.0:
            raise ValueError("key_prior_strength must be positive")
        if self.minimum_key_variance <= 0.0:
            raise ValueError("minimum_key_variance must be positive")
        if self.maximum_key_variance < self.minimum_key_variance:
            raise ValueError(
                "maximum_key_variance must be at least minimum_key_variance"
            )
        self.key_weight = np.zeros(self.max_neurons, dtype=np.float64)
        self.key_m2 = np.zeros(
            (self.max_neurons, self.input_size), dtype=np.float64
        )
        self.key_variance = np.full(
            (self.max_neurons, self.input_size),
            np.clip(
                self.rbf_width**2,
                self.minimum_key_variance,
                self.maximum_key_variance,
            ),
            dtype=np.float64,
        )

    def _activities(self, features: FloatArray) -> FloatArray:
        activities = np.zeros(self.max_neurons, dtype=np.float64)
        count = int(self.active_count[0])
        if count == 0:
            return activities
        differences = self.neuron_centers[:count] - features
        scaled_distance = np.mean(
            np.square(differences) / self.key_variance[:count], axis=1
        )
        activities[:count] = np.exp(-0.5 * scaled_distance)
        return activities

    def _recruit(self, features: FloatArray, target_class: int) -> bool:
        index = int(self.active_count[0])
        activated = super()._recruit(features, target_class)
        initial_variance = float(
            np.clip(
                self.rbf_width**2,
                self.minimum_key_variance,
                self.maximum_key_variance,
            )
        )
        self.key_weight[index] = self.key_prior_strength
        self.key_m2[index].fill(initial_variance * self.key_prior_strength)
        self.key_variance[index].fill(initial_variance)
        return activated

    def _update_neuron_statistics(
        self,
        features: FloatArray,
        activities: FloatArray,
        active_before_update: int,
    ) -> None:
        super()._update_neuron_statistics(
            features, activities, active_before_update
        )
        if active_before_update == 0:
            return
        active_slice = slice(0, active_before_update)
        activation = activities[active_slice]
        new_weight = self.key_weight[active_slice] + activation
        delta = features - self.neuron_centers[active_slice]
        self.neuron_centers[active_slice] += (
            activation[:, None] * delta / new_weight[:, None]
        )
        delta_after = features - self.neuron_centers[active_slice]
        self.key_m2[active_slice] += activation[:, None] * delta * delta_after
        self.key_weight[active_slice] = new_weight
        variance = self.key_m2[active_slice] / new_weight[:, None]
        self.key_variance[active_slice] = np.clip(
            variance,
            self.minimum_key_variance,
            self.maximum_key_variance,
        )

    @property
    def diagnostics(self) -> dict[str, float | int | bool]:
        result = super().diagnostics
        count = int(self.active_count[0])
        result.update(
            {
                "adaptive_keys": True,
                "mean_key_evidence": (
                    0.0 if count == 0 else float(np.mean(self.key_weight[:count]))
                ),
                "mean_learned_width": (
                    0.0
                    if count == 0
                    else float(np.mean(np.sqrt(self.key_variance[:count])))
                ),
                "minimum_learned_width": (
                    0.0
                    if count == 0
                    else float(np.min(np.sqrt(self.key_variance[:count])))
                ),
                "maximum_learned_width": (
                    0.0
                    if count == 0
                    else float(np.max(np.sqrt(self.key_variance[:count])))
                ),
                "mean_value_norm": (
                    0.0
                    if count == 0
                    else float(
                        np.mean(
                            np.linalg.norm(
                                self.expanded_weights[
                                    :, self.input_size : self.input_size + count
                                ],
                                axis=0,
                            )
                        )
                    )
                ),
            }
        )
        return result

    @property
    def state_nbytes(self) -> int:
        return super().state_nbytes + sum(
            array.nbytes
            for array in (self.key_weight, self.key_m2, self.key_variance)
        )


@dataclass
class ProtectedFastSlowReadout:
    """Fast decaying adapter over a non-overwriting prototype memory."""

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
        self.fast_weights = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )

    @property
    def weights(self) -> FloatArray:
        return self.slow_memory.centroids + self.fast_scale * self.fast_weights

    def predict(self, features: FloatArray) -> FloatArray:
        features = _validate_vector("features", features, self.input_size)
        return self.slow_memory.predict(features) + self.fast_scale * (
            self.fast_weights @ features
        )

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        self.fast_weights *= self.fast_decay
        error = target - prediction
        scale = self.fast_learning_rate / (
            self.epsilon + float(features @ features)
        )
        update = scale * np.outer(error, features)
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.fast_weights += update
        self.slow_memory.update(features, target, prediction)

    @property
    def state_nbytes(self) -> int:
        return self.slow_memory.state_nbytes + self.fast_weights.nbytes


@dataclass
class FastSlowLMSReadout:
    """Online readout with quickly adapting and slowly consolidated weights."""

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
        self.slow_weights = np.zeros(
            (self.output_size, self.input_size), dtype=np.float64
        )
        self.fast_weights = np.zeros_like(self.slow_weights)

    @property
    def weights(self) -> FloatArray:
        return self.slow_weights + self.fast_weights

    def predict(self, features: FloatArray) -> FloatArray:
        features = _validate_vector("features", features, self.input_size)
        return self.weights @ features

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None:
        features = _validate_vector("features", features, self.input_size)
        target = _validate_vector("target", target, self.output_size)
        prediction = _validate_vector("prediction", prediction, self.output_size)
        self.fast_weights *= self.fast_decay
        error = target - prediction
        scale = self.learning_rate / (self.epsilon + float(features @ features))
        update = scale * np.outer(error, features)
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.fast_weights += update
        transfer = self.consolidation_rate * self.fast_weights
        self.slow_weights += transfer
        self.fast_weights -= transfer

    @property
    def state_nbytes(self) -> int:
        return self.slow_weights.nbytes + self.fast_weights.nbytes

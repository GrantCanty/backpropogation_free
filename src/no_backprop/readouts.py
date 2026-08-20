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

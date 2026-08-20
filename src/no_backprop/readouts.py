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

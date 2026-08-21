"""Exact and structured recursive least-squares baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.frozen import FrozenReadout
from continual_core.protocols import FloatArray
from continual_core.validation import vector


def _validate_parameters(regularization: float, forgetting_factor: float) -> None:
    if regularization <= 0.0:
        raise ValueError("regularization must be positive")
    if not 0.0 < forgetting_factor <= 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")


@dataclass
class RLSReadout(FrozenReadout):
    regularization: float = 1.0
    forgetting_factor: float = 0.999

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_parameters(self.regularization, self.forgetting_factor)
        self.inverse_correlation = (
            np.eye(self.input_size, dtype=np.float64) / self.regularization
        )

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        projected = self.inverse_correlation @ features
        denominator = self.forgetting_factor + float(features @ projected)
        gain = projected / denominator
        self.weights += np.outer(target - prediction, gain)
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

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.weights, self.inverse_correlation)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            "inverse_correlation": self.inverse_correlation,
        }


@dataclass
class DiagonalRLSReadout(FrozenReadout):
    """Linear-memory approximation retaining only the inverse diagonal."""

    regularization: float = 1.0
    forgetting_factor: float = 0.999

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_parameters(self.regularization, self.forgetting_factor)
        self.inverse_diagonal = np.full(
            self.input_size, 1.0 / self.regularization, dtype=np.float64
        )

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        projected = self.inverse_diagonal * features
        denominator = self.forgetting_factor + float(features @ projected)
        gain = projected / denominator
        self.weights += np.outer(target - prediction, gain)
        self.inverse_diagonal = (
            self.inverse_diagonal - np.square(projected) / denominator
        ) / self.forgetting_factor
        np.maximum(
            self.inverse_diagonal,
            np.finfo(float).tiny,
            out=self.inverse_diagonal,
        )

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes + self.inverse_diagonal.nbytes

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.weights, self.inverse_diagonal)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            "inverse_diagonal": self.inverse_diagonal,
        }


@dataclass
class BlockRLSReadout(FrozenReadout):
    """Approximation preserving correlations within fixed feature blocks."""

    regularization: float = 1.0
    forgetting_factor: float = 0.999
    block_size: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_parameters(self.regularization, self.forgetting_factor)
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
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        projected = [
            inverse @ features[block]
            for block, inverse in zip(self.block_slices, self.inverse_blocks)
        ]
        denominator = self.forgetting_factor + sum(
            float(features[block] @ item)
            for block, item in zip(self.block_slices, projected)
        )
        self.weights += np.outer(
            target - prediction, np.concatenate(projected) / denominator
        )
        for index, (inverse, item) in enumerate(
            zip(self.inverse_blocks, projected)
        ):
            updated = (
                inverse - np.outer(item, item) / denominator
            ) / self.forgetting_factor
            self.inverse_blocks[index] = 0.5 * (updated + updated.T)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes + sum(x.nbytes for x in self.inverse_blocks)

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.weights, *self.inverse_blocks)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            **{
                f"inverse_block_{index}": block
                for index, block in enumerate(self.inverse_blocks)
            },
        }

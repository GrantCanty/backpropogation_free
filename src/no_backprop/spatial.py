"""Fixed-width, non-recurrent image frontends for online readouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from no_backprop.protocol import FloatArray, ProtocolError
from no_backprop.readouts import Readout


def representation_statistics(matrix: FloatArray) -> dict[str, float | int]:
    """Measure variance and spectral rank without retaining data in the model."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError("representations must be a non-empty matrix")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(matrix) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(np.sum(eigenvalues))
    if total <= np.finfo(float).tiny:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
    return {
        "samples": len(matrix),
        "feature_width": matrix.shape[1],
        "effective_rank": effective_rank,
        "normalized_effective_rank": effective_rank / matrix.shape[1],
        "mean_feature_variance": float(np.mean(np.diag(covariance))),
        "mean_representation_norm": float(
            np.mean(np.linalg.norm(matrix, axis=1))
        ),
    }


class ImageEncoder(Protocol):
    """Stateless image-to-vector transform used before an online readout."""

    name: str
    output_size: int
    persistent_arrays: tuple[np.ndarray, ...]

    def encode(self, image: FloatArray) -> FloatArray: ...


@dataclass
class FlattenedImageEncoder:
    """Expose all pixels once, preserving the input's fixed spatial order."""

    image_size: int = 8

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.name = "flattened_pixels"
        self.output_size = self.image_size * self.image_size
        self.persistent_arrays: tuple[np.ndarray, ...] = ()

    def encode(self, image: FloatArray) -> FloatArray:
        image = np.asarray(image, dtype=np.float64)
        expected = (self.image_size, self.image_size)
        if image.shape != expected:
            raise ValueError(f"image must have shape {expected}")
        return image.reshape(-1).copy()


@dataclass
class FixedConvolutionImageEncoder:
    """Four fixed orthogonal 3x3 filters followed by 2x2 average pooling."""

    image_size: int = 8
    filters: int = 4
    kernel_size: int = 3
    pool_size: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.filters <= 0:
            raise ValueError("filters must be positive")
        if self.filters > self.kernel_size * self.kernel_size - 1:
            raise ValueError("too many zero-mean orthogonal filters")
        if self.pool_size <= 0 or self.image_size % self.pool_size:
            raise ValueError("pool_size must evenly divide image_size")

        rng = np.random.default_rng(self.seed)
        raw = rng.normal(
            size=(self.kernel_size * self.kernel_size, self.filters)
        )
        raw -= np.mean(raw, axis=0, keepdims=True)
        orthogonal, _ = np.linalg.qr(raw)
        self.kernels = orthogonal.T.reshape(
            self.filters, self.kernel_size, self.kernel_size
        )
        pooled_size = self.image_size // self.pool_size
        self.name = "fixed_convolution"
        self.output_size = self.filters * pooled_size * pooled_size
        self.persistent_arrays = (self.kernels,)

    def encode(self, image: FloatArray) -> FloatArray:
        return self.feature_map(image).reshape(-1)

    def feature_map(self, image: FloatArray) -> FloatArray:
        """Return the pooled channel-first map before vectorization."""

        image = np.asarray(image, dtype=np.float64)
        expected = (self.image_size, self.image_size)
        if image.shape != expected:
            raise ValueError(f"image must have shape {expected}")
        padding = self.kernel_size // 2
        padded = np.pad(image, padding, mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (self.kernel_size, self.kernel_size)
        )
        activations = np.tanh(
            np.einsum("hwkl,fkl->fhw", windows, self.kernels)
        )
        pooled_size = self.image_size // self.pool_size
        pooled = activations.reshape(
            self.filters,
            pooled_size,
            self.pool_size,
            pooled_size,
            self.pool_size,
        ).mean(axis=(2, 4))
        # Undo the RMS reduction caused by averaging independent 2x2 values.
        pooled *= np.sqrt(float(self.pool_size * self.pool_size))
        return pooled


@dataclass
class PolarityConvolutionImageEncoder:
    """Fixed convolution with an explicit contrast-polarity representation."""

    image_size: int = 8
    mode: Literal["absolute", "signed_magnitude"] = "absolute"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("absolute", "signed_magnitude"):
            raise ValueError(f"unknown polarity mode: {self.mode}")
        filters = 4 if self.mode == "absolute" else 2
        self.base = FixedConvolutionImageEncoder(
            image_size=self.image_size,
            filters=filters,
            seed=self.seed,
        )
        self.name = f"{self.mode}_convolution"
        self.output_size = (
            self.base.output_size
            if self.mode == "absolute"
            else 2 * self.base.output_size
        )
        self.persistent_arrays = self.base.persistent_arrays

    @property
    def kernels(self) -> FloatArray:
        return self.base.kernels

    def encode(self, image: FloatArray) -> FloatArray:
        signed = self.base.encode(image)
        if self.mode == "absolute":
            return np.abs(signed)
        return np.concatenate((signed, np.abs(signed)))


@dataclass(frozen=True)
class SpatialClassifierConfig:
    image_size: int = 8
    output_size: int = 10
    frontend: Literal[
        "pixels",
        "fixed_convolution",
        "absolute_convolution",
        "signed_magnitude_convolution",
    ] = "pixels"
    seed: int = 0


class OnlineSpatialClassifier:
    """Predict-before-learn wrapper that consumes one complete image at a time."""

    def __init__(self, config: SpatialClassifierConfig, readout: Readout) -> None:
        if config.frontend == "pixels":
            encoder: ImageEncoder = FlattenedImageEncoder(config.image_size)
        elif config.frontend == "fixed_convolution":
            encoder = FixedConvolutionImageEncoder(
                image_size=config.image_size, seed=config.seed
            )
        elif config.frontend in (
            "absolute_convolution",
            "signed_magnitude_convolution",
        ):
            mode = (
                "absolute"
                if config.frontend == "absolute_convolution"
                else "signed_magnitude"
            )
            encoder = PolarityConvolutionImageEncoder(
                image_size=config.image_size,
                mode=mode,
                seed=config.seed,
            )
        else:  # pragma: no cover - Literal guards typed callers
            raise ValueError(f"unknown spatial frontend: {config.frontend}")
        expected_features = encoder.output_size + 1
        if readout.input_size != expected_features:
            raise ValueError(
                f"readout input_size must be encoder output + 1 ({expected_features})"
            )
        if readout.output_size != config.output_size:
            raise ValueError("readout output_size does not match classifier")

        self.config = config
        self.encoder = encoder
        self.readout = readout
        # These names intentionally match OnlineReservoir's persistent/transient
        # interface so evaluation locking and checkpoints cover both learners.
        self.input_weights = (
            encoder.persistent_arrays[0]
            if encoder.persistent_arrays
            else np.empty(0, dtype=np.float64)
        )
        self.recurrent_weights = np.empty(0, dtype=np.float64)
        self.bias = np.empty(0, dtype=np.float64)
        self.state = np.empty(0, dtype=np.float64)
        self._pending_features: FloatArray | None = None
        self._pending_prediction: FloatArray | None = None

    def predict(self, image: FloatArray) -> FloatArray:
        if self._pending_prediction is not None:
            raise ProtocolError("learn must be called before the next prediction")
        encoded = self.encoder.encode(image)
        features = np.concatenate((encoded, np.ones(1, dtype=np.float64)))
        prediction = self.readout.predict(features)
        self._pending_features = features
        self._pending_prediction = prediction.copy()
        return prediction.copy()

    def learn(self, target: FloatArray) -> FloatArray:
        if self._pending_prediction is None or self._pending_features is None:
            raise ProtocolError("predict must be called before learn")
        target = np.asarray(target, dtype=np.float64)
        expected = (self.config.output_size,)
        if target.shape != expected:
            raise ValueError(f"target must have shape {expected}")
        prediction = self._pending_prediction
        error = target - prediction
        if np.all(np.isfinite(target)):
            self.readout.update(self._pending_features, target, prediction)
        self._pending_features = None
        self._pending_prediction = None
        return error.copy()

    def reset_state(self) -> None:
        if self._pending_prediction is not None:
            raise ProtocolError("cannot reset between predict and learn")

    def representation_diagnostics(
        self, images: FloatArray
    ) -> dict[str, float | int]:
        matrix = np.stack([self.encoder.encode(image) for image in images])
        return representation_statistics(matrix)

    @property
    def diagnostics(self) -> dict[str, str | int | bool]:
        return {
            "frontend": self.encoder.name,
            "recurrent": False,
            "image_events_per_prediction": 1,
            "feature_width": self.encoder.output_size,
            "fixed_frontend": True,
        }

    @property
    def state_nbytes(self) -> int:
        return (
            sum(array.nbytes for array in self.encoder.persistent_arrays)
            + self.readout.state_nbytes
        )

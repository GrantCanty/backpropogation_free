"""Forward-only masked prediction in a fixed convolutional representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from no_backprop.protocol import FloatArray, ProtocolError
from no_backprop.readouts import RLSReadout, Readout
from no_backprop.spatial import (
    FixedConvolutionImageEncoder,
    representation_statistics,
)


@dataclass(frozen=True)
class PredictiveSpatialConfig:
    """Four 2x2 target blocks over a 4x4 fixed-convolution feature map."""

    image_size: int = 8
    output_size: int = 10
    predictor_regularization: float = 1.0
    seed: int = 0


class OnlinePredictiveSpatialClassifier:
    """JEPA-inspired latent predictor followed by one online classifier.

    The fixed convolutional map is split into four quadrants. For each target
    quadrant, a cumulative RLS predictor sees the other three quadrants and a
    one-hot target position. Its four predictions form the 64-coordinate image
    representation consumed by the classifier. Both learning stages happen
    only after the pre-update classification prediction has been returned.
    """

    block_starts = ((0, 0), (0, 2), (2, 0), (2, 2))

    def __init__(self, config: PredictiveSpatialConfig, readout: Readout) -> None:
        if config.image_size != 8:
            raise ValueError("the initial predictive frontend requires 8x8 images")
        if config.predictor_regularization <= 0.0:
            raise ValueError("predictor_regularization must be positive")
        self.config = config
        self.encoder = FixedConvolutionImageEncoder(
            image_size=config.image_size, seed=config.seed
        )
        if self.encoder.output_size != 64:
            raise ValueError("predictive encoder must emit 64 target coordinates")
        if readout.input_size != 65:
            raise ValueError("classifier readout must consume 64 features plus bias")
        if readout.output_size != config.output_size:
            raise ValueError("readout output_size does not match classifier")

        self.readout = readout
        self.context_size = 64 + len(self.block_starts) + 1
        self.target_size = self.encoder.filters * 2 * 2
        self.predictor = RLSReadout(
            self.context_size,
            self.target_size,
            seed=config.seed + 101,
            regularization=config.predictor_regularization,
            forgetting_factor=1.0,
        )
        self.predictor_image_count = np.zeros(1, dtype=np.float64)
        self.predictor_update_count = np.zeros(1, dtype=np.float64)
        self.predictor_squared_error_sum = np.zeros(1, dtype=np.float64)

        # Match the reservoir/spatial learner interface used by checkpointing
        # and locked evaluation. The convolution kernels are fixed model state.
        self.input_weights = self.encoder.kernels
        self.recurrent_weights = np.empty(0, dtype=np.float64)
        self.bias = np.empty(0, dtype=np.float64)
        self.state = np.empty(0, dtype=np.float64)
        self._pending_features: FloatArray | None = None
        self._pending_prediction: FloatArray | None = None
        self._pending_contexts: FloatArray | None = None
        self._pending_targets: FloatArray | None = None
        self._pending_target_predictions: FloatArray | None = None

    def _context_target_pairs(
        self, image: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        feature_map = self.encoder.feature_map(image)
        contexts = np.zeros(
            (len(self.block_starts), self.context_size), dtype=np.float64
        )
        targets = np.zeros(
            (len(self.block_starts), self.target_size), dtype=np.float64
        )
        for index, (row, column) in enumerate(self.block_starts):
            masked = feature_map.copy()
            target = masked[:, row : row + 2, column : column + 2].copy()
            masked[:, row : row + 2, column : column + 2] = 0.0
            position = np.zeros(len(self.block_starts), dtype=np.float64)
            position[index] = 1.0
            contexts[index] = np.concatenate(
                (masked.reshape(-1), position, np.ones(1, dtype=np.float64))
            )
            targets[index] = target.reshape(-1)
        return contexts, targets

    def _predict_representation(
        self, contexts: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        target_predictions = np.stack(
            [self.predictor.predict(context) for context in contexts]
        )
        predicted_map = np.zeros((self.encoder.filters, 4, 4), dtype=np.float64)
        for prediction, (row, column) in zip(
            target_predictions, self.block_starts
        ):
            predicted_map[:, row : row + 2, column : column + 2] = (
                prediction.reshape(self.encoder.filters, 2, 2)
            )
        return predicted_map.reshape(-1), target_predictions

    def encode_prediction(self, image: FloatArray) -> FloatArray:
        """Return the current predicted representation without updating state."""

        contexts, _ = self._context_target_pairs(image)
        representation, _ = self._predict_representation(contexts)
        return representation

    def predict(self, image: FloatArray) -> FloatArray:
        if self._pending_prediction is not None:
            raise ProtocolError("learn must be called before the next prediction")
        contexts, targets = self._context_target_pairs(image)
        representation, target_predictions = self._predict_representation(
            contexts
        )
        features = np.concatenate(
            (representation, np.ones(1, dtype=np.float64))
        )
        prediction = self.readout.predict(features)
        self._pending_features = features
        self._pending_prediction = prediction.copy()
        self._pending_contexts = contexts
        self._pending_targets = targets
        self._pending_target_predictions = target_predictions
        return prediction.copy()

    def learn(self, target: FloatArray) -> FloatArray:
        if (
            self._pending_prediction is None
            or self._pending_features is None
            or self._pending_contexts is None
            or self._pending_targets is None
            or self._pending_target_predictions is None
        ):
            raise ProtocolError("predict must be called before learn")
        target = np.asarray(target, dtype=np.float64)
        expected = (self.config.output_size,)
        if target.shape != expected:
            raise ValueError(f"target must have shape {expected}")
        prediction = self._pending_prediction
        error = target - prediction
        if np.all(np.isfinite(target)):
            # The classifier sees the pre-update predictive representation.
            self.readout.update(self._pending_features, target, prediction)
            for context, block, block_prediction in zip(
                self._pending_contexts,
                self._pending_targets,
                self._pending_target_predictions,
            ):
                self.predictor.update(context, block, block_prediction)
                self.predictor_squared_error_sum[0] += float(
                    np.mean(np.square(block - block_prediction))
                )
                self.predictor_update_count[0] += 1.0
            self.predictor_image_count[0] += 1.0
        self._clear_pending()
        return error.copy()

    def _clear_pending(self) -> None:
        self._pending_features = None
        self._pending_prediction = None
        self._pending_contexts = None
        self._pending_targets = None
        self._pending_target_predictions = None

    def reset_state(self) -> None:
        if self._pending_prediction is not None:
            raise ProtocolError("cannot reset between predict and learn")

    def representation_diagnostics(
        self, images: FloatArray
    ) -> dict[str, float | int]:
        images = np.asarray(images, dtype=np.float64)
        representations: list[FloatArray] = []
        losses: list[float] = []
        for image in images:
            contexts, targets = self._context_target_pairs(image)
            representation, predictions = self._predict_representation(contexts)
            representations.append(representation)
            losses.append(float(np.mean(np.square(targets - predictions))))
        matrix = np.stack(representations)
        result = representation_statistics(matrix)
        result["target_prediction_mse"] = float(np.mean(losses))
        return result

    @property
    def diagnostics(self) -> dict[str, str | int | float | bool]:
        updates = self.predictor_update_count[0]
        return {
            "frontend": "fixed_convolution_predictive_blocks",
            "recurrent": False,
            "image_events_per_prediction": 1,
            "feature_width": 64,
            "fixed_frontend": True,
            "jepa_inspired": True,
            "target_blocks": len(self.block_starts),
            "target_block_shape": "2x2x4",
            "predictor": "cumulative_rls",
            "predictor_forgetting_factor": 1.0,
            "predictor_images": int(self.predictor_image_count[0]),
            "predictor_updates": int(updates),
            "mean_online_target_prediction_mse": (
                0.0
                if updates == 0.0
                else float(self.predictor_squared_error_sum[0] / updates)
            ),
            "stored_raw_samples": 0,
        }

    @property
    def state_nbytes(self) -> int:
        counters = (
            self.predictor_image_count,
            self.predictor_update_count,
            self.predictor_squared_error_sum,
        )
        return (
            self.encoder.kernels.nbytes
            + self.predictor.state_nbytes
            + sum(array.nbytes for array in counters)
            + self.readout.state_nbytes
        )

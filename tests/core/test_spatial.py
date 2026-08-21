import numpy as np

from continual_core.evaluation import PredictLearnAdapter, evaluate_classification_locked
from continual_core.spatial import (
    FixedConvolutionImageEncoder,
    FlattenedImageEncoder,
    OnlineSpatialClassifier,
    PolarityConvolutionImageEncoder,
    SpatialClassifierConfig,
)


class LinearReadout:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size), dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features

    def update(
        self,
        features: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        self.weights += np.outer(target - prediction, features)

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (self.weights,)

    @property
    def state_nbytes(self) -> int:
        return self.weights.nbytes


def target(label: int) -> np.ndarray:
    value = np.zeros(2)
    value[label] = 1.0
    return value


def test_matched_frontends_emit_exactly_64_features() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    pixels = FlattenedImageEncoder()
    convolution = FixedConvolutionImageEncoder(seed=5)
    np.testing.assert_array_equal(pixels.encode(image), image.reshape(-1))
    assert pixels.encode(image).shape == convolution.encode(image).shape == (64,)


def test_fixed_convolution_filters_are_deterministic_orthogonal_and_zero_mean() -> None:
    left = FixedConvolutionImageEncoder(seed=7)
    right = FixedConvolutionImageEncoder(seed=7)
    other = FixedConvolutionImageEncoder(seed=8)
    np.testing.assert_array_equal(left.kernels, right.kernels)
    assert not np.array_equal(left.kernels, other.kernels)
    flattened = left.kernels.reshape(4, -1)
    np.testing.assert_allclose(flattened @ flattened.T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(np.mean(flattened, axis=1), 0.0, atol=1e-12)


def test_polarity_convolutions_preserve_width_and_expected_channels() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    absolute = PolarityConvolutionImageEncoder(mode="absolute", seed=7)
    signed_magnitude = PolarityConvolutionImageEncoder(
        mode="signed_magnitude", seed=7
    )
    assert absolute.encode(image).shape == signed_magnitude.encode(image).shape == (64,)
    assert np.all(absolute.encode(image) >= 0.0)
    np.testing.assert_array_equal(
        signed_magnitude.encode(image)[32:],
        np.abs(signed_magnitude.encode(image)[:32]),
    )


def test_spatial_classifier_uses_generic_readout_and_locked_evaluation() -> None:
    learner = OnlineSpatialClassifier(
        SpatialClassifierConfig(output_size=2, frontend="pixels"),
        LinearReadout(65, 2),
    )
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    prediction = learner.predict(image)
    learner.learn(target(1))
    assert prediction.shape == (2,)
    result = evaluate_classification_locked(
        learner,
        (image,),
        (1,),
        PredictLearnAdapter(),
        target,
    )
    assert result["weights_unchanged"]

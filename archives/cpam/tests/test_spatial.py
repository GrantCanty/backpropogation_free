import numpy as np
import pytest

from no_backprop.checkpoint import restore_checkpoint, save_checkpoint
from no_backprop.experiment import (
    DigitsExperimentConfig,
    _digit_target,
    _evaluate_digits_locked,
    _process_digit_image,
    build_digits_learner,
)
from no_backprop.frontend_comparison import (
    FrontendComparisonConfig,
    PolarityComparisonConfig,
    run_frontend_comparison,
    run_polarity_comparison,
)
from no_backprop.spatial import (
    FixedConvolutionImageEncoder,
    FlattenedImageEncoder,
    OnlineSpatialClassifier,
    PolarityConvolutionImageEncoder,
)


def test_matched_frontends_emit_exactly_64_features() -> None:
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    pixels = FlattenedImageEncoder()
    convolution = FixedConvolutionImageEncoder(seed=5)

    np.testing.assert_array_equal(pixels.encode(image), image.reshape(-1))
    assert pixels.encode(image).shape == (64,)
    assert convolution.encode(image).shape == (64,)
    assert np.all(np.isfinite(convolution.encode(image)))


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

    absolute_features = absolute.encode(image)
    signed_magnitude_features = signed_magnitude.encode(image)
    assert absolute_features.shape == signed_magnitude_features.shape == (64,)
    assert np.all(absolute_features >= 0.0)
    np.testing.assert_array_equal(
        signed_magnitude_features[32:],
        np.abs(signed_magnitude_features[:32]),
    )
    assert signed_magnitude.feature_map(image).shape == (4, 4, 4)
    np.testing.assert_array_equal(
        signed_magnitude.feature_map(image).reshape(-1),
        signed_magnitude_features,
    )


@pytest.mark.parametrize(
    "kind",
    (
        "managed16_pixels",
        "managed16_fixed_conv",
        "managed16_absolute_conv",
        "managed16_signed_magnitude_conv",
    ),
)
def test_spatial_frontend_uses_one_image_event_and_locked_evaluation(kind: str) -> None:
    config = DigitsExperimentConfig(hidden_size=64, seed=11)
    learner = build_digits_learner(kind, config)
    assert isinstance(learner, OnlineSpatialClassifier)
    image = np.linspace(0.0, 1.0, 64).reshape(8, 8)

    _process_digit_image(learner, image, target=_digit_target(3))
    assert learner.readout.diagnostics["samples_in_cumulative_statistics"] == 1
    evaluation = _evaluate_digits_locked(
        learner,
        np.stack((image, 1.0 - image)),
        np.array([3, 4]),
    )
    assert evaluation["weights_unchanged"]
    assert evaluation["transient_state_restored"]
    assert learner.readout.diagnostics["samples_in_cumulative_statistics"] == 1


def test_spatial_frontends_match_readout_width_and_bound_state() -> None:
    config = DigitsExperimentConfig(hidden_size=64, seed=13)
    recurrent = build_digits_learner("probation_managed16", config)
    pixels = build_digits_learner("managed16_pixels", config)
    convolution = build_digits_learner("managed16_fixed_conv", config)

    assert recurrent.readout.input_size == pixels.readout.input_size == 65
    assert convolution.readout.input_size == 65
    assert convolution.state_nbytes - pixels.state_nbytes == 4 * 3 * 3 * 8
    before = convolution.state_nbytes
    _process_digit_image(
        convolution, np.ones((8, 8)), target=_digit_target(1)
    )
    assert convolution.state_nbytes == before


@pytest.mark.parametrize(
    "kind",
    (
        "managed16_pixels",
        "managed16_fixed_conv",
        "managed16_absolute_conv",
        "managed16_signed_magnitude_conv",
    ),
)
def test_spatial_frontend_checkpoint_round_trip(tmp_path, kind: str) -> None:
    config = DigitsExperimentConfig(hidden_size=64, seed=17)
    learner = build_digits_learner(kind, config)
    rng = np.random.default_rng(17)
    for label in (1, 4, 1, 8):
        _process_digit_image(
            learner, rng.uniform(size=(8, 8)), target=_digit_target(label)
        )
    path = save_checkpoint(learner, tmp_path / f"{kind}.npz")
    evaluation_image = rng.uniform(size=(8, 8))
    expected = _process_digit_image(learner, evaluation_image, target=None)

    restored = build_digits_learner(kind, config)
    restore_checkpoint(restored, path)
    actual = _process_digit_image(restored, evaluation_image, target=None)
    np.testing.assert_allclose(actual, expected)
    assert restored.readout.diagnostics == learner.readout.diagnostics


def test_spatial_frontends_reject_unmatched_hidden_width() -> None:
    with pytest.raises(ValueError, match="hidden_size=64"):
        build_digits_learner(
            "managed16_pixels", DigitsExperimentConfig(hidden_size=32)
        )


def test_frontend_comparison_reports_matched_invariants() -> None:
    result = run_frontend_comparison(
        FrontendComparisonConfig(
            seeds=(5,),
            test_per_class=2,
            protocols=("shuffled",),
            include_drift=True,
        )
    )
    assert result["dataset"]["downloaded_data_bytes"] == 0
    assert result["invariants"]["same_downstream_memory"]
    assert result["invariants"]["same_readout_feature_width"]
    assert result["invariants"]["one_label_update_per_image"]
    assert result["invariants"]["weights_locked_during_evaluation"]
    assert result["invariants"]["bounded_state"]
    assert result["invariants"]["raw_samples_stored"] == 0
    assert set(result["summary"]["quality"]["shuffled"]) == {
        "probation_managed16",
        "managed16_pixels",
        "managed16_fixed_conv",
    }
    assert set(result["summary"]["drift"]) == {
        "probation_managed16",
        "managed16_pixels",
        "managed16_fixed_conv",
    }
    assert set(result["summary"]["paired_drift_difference_from_recurrent"]) == {
        "managed16_pixels",
        "managed16_fixed_conv",
    }


def test_polarity_comparison_moves_signed_magnitude_cosine_toward_recurrence() -> None:
    result = run_polarity_comparison(
        PolarityComparisonConfig(
            seeds=(5,),
            test_per_class=4,
            protocols=("shuffled",),
            include_drift=False,
        )
    )
    geometry = result["summary"]["inversion_geometry"]
    recurrent = geometry["probation_managed16"][
        "mean_original_inverted_cosine"
    ]["mean"]
    fixed = geometry["managed16_fixed_conv"][
        "mean_original_inverted_cosine"
    ]["mean"]
    signed_magnitude = geometry["managed16_signed_magnitude_conv"][
        "mean_original_inverted_cosine"
    ]["mean"]
    absolute = geometry["managed16_absolute_conv"][
        "mean_original_inverted_cosine"
    ]["mean"]

    assert abs(signed_magnitude - recurrent) < abs(fixed - recurrent)
    assert absolute > signed_magnitude > fixed
    assert result["invariants"]["all_geometry_feature_widths_are_64"]
    assert result["invariants"]["geometry_measured_without_learning"]

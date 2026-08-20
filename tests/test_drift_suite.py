import numpy as np
import pytest

import no_backprop.drift_suite as drift_suite
from no_backprop.digits import DigitsSplit
from no_backprop.drift_suite import (
    DriftSuiteConfig,
    PRIMARY_SPATIAL_BASELINE,
    run_drift_suite,
    transform_images,
    transformed_split,
)


def test_drift_transformations_are_deterministic_bounded_and_distinct() -> None:
    config = DriftSuiteConfig(seeds=(3,))
    rng = np.random.default_rng(3)
    images = rng.uniform(size=(5, 8, 8))
    original = images.copy()
    outputs = {}
    for name in config.transformations:
        first = transform_images(images, name, config, seed=31)
        second = transform_images(images, name, config, seed=31)
        np.testing.assert_array_equal(first, second)
        assert first.shape == images.shape
        assert np.all((0.0 <= first) & (first <= 1.0))
        assert not np.array_equal(first, images)
        outputs[name] = first
    np.testing.assert_array_equal(images, original)
    assert len({output.tobytes() for output in outputs.values()}) == len(outputs)


def test_drift_transformations_have_expected_semantics() -> None:
    config = DriftSuiteConfig(seeds=(5,))
    image = np.arange(64, dtype=np.float64).reshape(1, 8, 8) / 63.0

    np.testing.assert_allclose(
        transform_images(image, "inversion", config, seed=1), 1.0 - image
    )
    low_contrast = transform_images(image, "low_contrast", config, seed=1)
    np.testing.assert_allclose(
        low_contrast, 0.5 + config.contrast_scale * (image - 0.5)
    )
    occluded = transform_images(image, "center_occlusion", config, seed=1)
    assert np.all(occluded[:, 3:5, 3:5] == 0.0)
    translated = transform_images(image, "translation", config, seed=1)
    np.testing.assert_array_equal(translated[:, 1:, 1:], image[:, :-1, :-1])
    assert np.all(translated[:, 0, :] == 0.0)
    assert np.all(translated[:, :, 0] == 0.0)


def test_transformed_split_preserves_labels_without_aliasing() -> None:
    rng = np.random.default_rng(7)
    split = DigitsSplit(
        train_images=rng.uniform(size=(8, 8, 8)),
        train_labels=np.arange(8) % 2,
        test_images=rng.uniform(size=(4, 8, 8)),
        test_labels=np.arange(4) % 2,
    )
    shifted = transformed_split(
        split,
        "gaussian_noise",
        DriftSuiteConfig(seeds=(7,)),
        seed=71,
    )

    np.testing.assert_array_equal(shifted.train_labels, split.train_labels)
    np.testing.assert_array_equal(shifted.test_labels, split.test_labels)
    assert not np.shares_memory(shifted.train_labels, split.train_labels)
    assert not np.shares_memory(shifted.test_labels, split.test_labels)
    assert not np.array_equal(shifted.train_images, split.train_images)


def test_drift_suite_uses_signed_magnitude_as_paired_baseline(monkeypatch) -> None:
    rng = np.random.default_rng(11)
    labels = np.arange(20) % 2
    split = DigitsSplit(
        train_images=rng.uniform(size=(20, 8, 8)),
        train_labels=labels,
        test_images=rng.uniform(size=(6, 8, 8)),
        test_labels=np.arange(6) % 2,
    )
    monkeypatch.setattr(
        drift_suite,
        "load_digits_split",
        lambda **_: split,
    )
    result = run_drift_suite(
        DriftSuiteConfig(
            seeds=(11,),
            test_per_class=1,
            transformations=("inversion", "translation"),
        )
    )

    assert result["baseline"] == PRIMARY_SPATIAL_BASELINE
    assert result["dataset"]["downloaded_data_bytes"] == 0
    assert result["dataset"]["labels_preserved_by_transformations"]
    invariants = result["invariants"]
    assert invariants["same_downstream_memory"]
    assert invariants["same_readout_feature_width"]
    assert invariants["one_label_update_per_image"]
    assert invariants["weights_locked_during_evaluation"]
    assert invariants["bounded_state"]
    assert invariants["original_phase_trained_once_per_model_and_reused"]
    assert invariants["state_reset_before_every_image"]
    assert invariants["raw_samples_stored_by_models"] == 0
    assert set(result["summary"]["by_transformation"]) == {
        "inversion",
        "translation",
    }
    paired = result["summary"][
        "paired_difference_from_signed_magnitude"
    ]["inversion"]
    assert PRIMARY_SPATIAL_BASELINE not in paired
    assert set(paired) == set(result["controls"])
    for run in result["runs"]:
        for model in run["models"].values():
            for transformation in model["transformations"].values():
                assert transformation["samples_in_cumulative_statistics"] == 60


def test_drift_suite_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="unique"):
        DriftSuiteConfig(seeds=(3, 3))
    with pytest.raises(ValueError, match="contrast_scale"):
        DriftSuiteConfig(seeds=(3,), contrast_scale=1.0)
    with pytest.raises(ValueError, match="signed-magnitude baseline"):
        DriftSuiteConfig(seeds=(3,), kinds=("managed16_fixed_conv",))
    with pytest.raises(ValueError, match="predictor_regularization"):
        DriftSuiteConfig(seeds=(3,), predictor_regularization=0.0)

import numpy as np
import pytest

from no_backprop.digits import (
    DigitsSplit,
    augment_digits_split,
    build_digits_segments,
    load_digits_split,
)
from no_backprop.experiment import DigitsExperimentConfig, run_digits_model


def test_bundled_digits_split_is_deterministic_and_stratified() -> None:
    pytest.importorskip("sklearn")
    left = load_digits_split(test_per_class=5, seed=4)
    right = load_digits_split(test_per_class=5, seed=4)
    np.testing.assert_array_equal(left.train_images, right.train_images)
    np.testing.assert_array_equal(left.test_labels, right.test_labels)
    assert left.train_images.shape[1:] == (8, 8)
    assert len(left.test_labels) == 50
    assert np.all(np.bincount(left.test_labels, minlength=10) == 5)
    assert 0.0 <= float(left.train_images.min())
    assert float(left.train_images.max()) <= 1.0


def test_shuffled_and_class_ordered_streams_cover_the_same_examples() -> None:
    labels = np.repeat(np.arange(3), 4)
    shuffled = build_digits_segments(labels, protocol="shuffled", seed=2)
    ordered = build_digits_segments(labels, protocol="class_ordered", seed=2)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([segment.indices for segment in shuffled])),
        np.arange(len(labels)),
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate([segment.indices for segment in ordered])),
        np.arange(len(labels)),
    )
    assert all(
        np.all(labels[segment.indices] == segment.focus_class) for segment in ordered
    )


def test_augmentation_is_deterministic_and_never_changes_test_data() -> None:
    rng = np.random.default_rng(5)
    split = DigitsSplit(
        train_images=rng.uniform(size=(6, 8, 8)),
        train_labels=np.arange(6),
        test_images=rng.uniform(size=(3, 8, 8)),
        test_labels=np.arange(3),
    )
    left = augment_digits_split(split, copies=1, seed=9)
    right = augment_digits_split(split, copies=1, seed=9)
    assert len(left.train_images) == 2 * len(split.train_images)
    np.testing.assert_array_equal(left.train_images, right.train_images)
    np.testing.assert_array_equal(left.test_images, split.test_images)
    np.testing.assert_array_equal(left.test_labels, split.test_labels)


@pytest.mark.parametrize("protocol", ["shuffled", "class_ordered"])
def test_digits_model_runs_prequentially_and_keeps_bounded_state(protocol: str) -> None:
    rng = np.random.default_rng(3)
    train_labels = np.repeat(np.arange(2), 6)
    test_labels = np.repeat(np.arange(2), 3)
    split = DigitsSplit(
        train_images=rng.uniform(size=(len(train_labels), 8, 8)),
        train_labels=train_labels,
        test_images=rng.uniform(size=(len(test_labels), 8, 8)),
        test_labels=test_labels,
    )
    result = run_digits_model(
        "lms",
        protocol,
        DigitsExperimentConfig(hidden_size=8, window=3, seed=3),
        split=split,
    )
    assert result["trained_samples"] == len(train_labels)
    assert result["bounded_state"]
    assert result["weights_locked_during_evaluation"]
    assert result["transient_state_restored_after_evaluation"]
    assert len(result["evaluation_history"]) == 3
    assert 0.0 <= result["final_test_accuracy"] <= 1.0

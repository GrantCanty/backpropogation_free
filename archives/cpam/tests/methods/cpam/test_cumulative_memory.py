import numpy as np
from continual_core.checkpoint import restore_checkpoint, save_checkpoint
from continual_core.datasets.digits import DigitsSplit
from experiments.legacy import (
    DigitsExperimentConfig,
    _digit_target,
    _process_digit_image,
    build_digits_learner,
    run_digits_model,
)
from methods.cpam.readouts import CumulativeMemoryReadout


def test_slow_memory_matches_batch_ridge_with_every_observation() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(40, 5))
    targets = rng.normal(size=(40, 3))
    regularization = 1.7
    readout = CumulativeMemoryReadout(
        5, 3, regularization=regularization
    )
    for feature, target in zip(features, targets):
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)

    expected = targets.T @ features @ np.linalg.inv(
        regularization * np.eye(5) + features.T @ features
    )
    np.testing.assert_allclose(readout.slow_weights, expected, atol=1e-10)
    assert readout.diagnostics["samples_in_slow_statistics"] == len(features)
    assert readout.diagnostics["stored_raw_samples"] == 0


def test_fast_memory_compresses_slow_errors_by_confusion() -> None:
    readout = CumulativeMemoryReadout(2, 2)
    observations = (
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    )
    for feature, target in observations:
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)

    # Both initial class-1 examples are compressed into the same (1 <- 0)
    # residual representation instead of being retained as two samples.
    assert readout.exception_counts[1, 0] == 2.0
    np.testing.assert_allclose(
        readout.exception_centroids[1, 0], np.array([0.5, 0.5])
    )
    assert readout.diagnostics["active_exception_representations"] == 1
    assert readout.diagnostics["stored_raw_samples"] == 0


def test_cumulative_memory_evaluation_is_locked_and_state_is_bounded() -> None:
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(2), 5)
    split = DigitsSplit(
        train_images=rng.uniform(size=(len(labels), 8, 8)),
        train_labels=labels,
        test_images=rng.uniform(size=(4, 8, 8)),
        test_labels=np.repeat(np.arange(2), 2),
    )
    result = run_digits_model(
        "cumulative_memory",
        "class_ordered",
        DigitsExperimentConfig(hidden_size=6, window=2, seed=3),
        split=split,
    )
    assert result["bounded_state"]
    assert result["weights_locked_during_evaluation"]
    assert result["memory_diagnostics"]["samples_in_slow_statistics"] == len(labels)
    assert result["memory_diagnostics"]["stored_raw_samples"] == 0


def test_cumulative_memory_checkpoint_round_trip(tmp_path) -> None:
    config = DigitsExperimentConfig(hidden_size=8, seed=5)
    learner = build_digits_learner("cumulative_memory", config)
    rng = np.random.default_rng(5)
    for label in (2, 7, 2):
        _process_digit_image(
            learner, rng.uniform(size=(8, 8)), target=_digit_target(label)
        )
    path = save_checkpoint(learner, tmp_path / "cumulative-memory.npz")
    evaluation_image = rng.uniform(size=(8, 8))
    expected = _process_digit_image(learner, evaluation_image, target=None)

    restored = build_digits_learner("cumulative_memory", config)
    restore_checkpoint(restored, path)
    actual = _process_digit_image(restored, evaluation_image, target=None)
    np.testing.assert_allclose(actual, expected)
    assert restored.readout.diagnostics == learner.readout.diagnostics


def test_factor_free_model_has_no_forgetting_or_decay_parameter() -> None:
    fields = CumulativeMemoryReadout.__dataclass_fields__
    assert "forgetting_factor" not in fields
    assert "decay" not in fields

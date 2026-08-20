import numpy as np
import pytest

from no_backprop.checkpoint import restore_checkpoint, save_checkpoint
from no_backprop.digits import DigitsSplit
from no_backprop.experiment import (
    DigitsExperimentConfig,
    _digit_target,
    _process_digit_image,
    build_digits_learner,
    run_digits_model,
)
from no_backprop.readouts import (
    CumulativeMaturityReadout,
    KeyValueMaturityReadout,
    _normalized_entropy,
)


def test_normalized_entropy_distinguishes_uniform_and_confident_scores() -> None:
    uniform = _normalized_entropy(np.zeros(3))
    confident = _normalized_entropy(np.array([8.0, -2.0, -3.0]))
    assert np.isclose(uniform, 1.0)
    assert 0.0 <= confident < 0.01


def test_maturity_without_neurons_matches_batch_ridge() -> None:
    rng = np.random.default_rng(14)
    features = rng.normal(size=(30, 4))
    targets = rng.normal(size=(30, 2))
    regularization = 1.3
    readout = CumulativeMaturityReadout(
        4, 2, regularization=regularization, max_neurons=0
    )
    for feature, target in zip(features, targets):
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)

    expected = targets.T @ features @ np.linalg.inv(
        regularization * np.eye(4) + features.T @ features
    )
    np.testing.assert_allclose(readout.weights, expected, atol=1e-10)
    assert readout.diagnostics["samples_in_cumulative_statistics"] == len(features)


def test_maturity_models_have_no_forgetting_or_decay_parameter() -> None:
    for readout_class in (CumulativeMaturityReadout, KeyValueMaturityReadout):
        fields = readout_class.__dataclass_fields__
        assert "forgetting_factor" not in fields
        assert "decay" not in fields
        assert "leverage_threshold" not in fields


def test_entropy_and_leverage_gates_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        CumulativeMaturityReadout(
            2, 2, entropy_gated=True, leverage_gated=True
        )


def test_entropy_blocks_uniform_startup_error_but_control_recruits() -> None:
    feature = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])
    control = CumulativeMaturityReadout(2, 2, max_neurons=2)
    entropy = CumulativeMaturityReadout(
        2, 2, max_neurons=2, entropy_gated=True
    )
    for readout in (control, entropy):
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)

    assert control.diagnostics["active_neurons"] == 1
    assert entropy.diagnostics["active_neurons"] == 0
    assert entropy.diagnostics["entropy_rejections"] == 1


def test_entropy_recruits_after_confident_error() -> None:
    readout = CumulativeMaturityReadout(
        2, 2, max_neurons=2, entropy_gated=True
    )
    feature = np.array([1.0, 0.0])
    correct_target = np.array([1.0, 0.0])
    changed_target = np.array([0.0, 1.0])

    prediction = readout.predict(feature)
    readout.update(feature, correct_target, prediction)
    prediction = readout.predict(feature)
    readout.update(feature, changed_target, prediction)

    assert readout.diagnostics["active_neurons"] == 1
    assert readout.neuron_recruitment_entropy[0] < 1.0


def test_leverage_gate_waits_until_an_error_region_is_familiar() -> None:
    readout = CumulativeMaturityReadout(
        2, 2, max_neurons=2, leverage_gated=True
    )
    feature = np.array([1.0, 0.0])
    first_target = np.array([0.0, 1.0])
    changed_target = np.array([1.0, 0.0])

    prediction = readout.predict(feature)
    readout.update(feature, first_target, prediction)
    assert readout.diagnostics["active_neurons"] == 0
    assert readout.diagnostics["leverage_rejections"] == 1

    prediction = readout.predict(feature)
    readout.update(feature, changed_target, prediction)
    assert readout.diagnostics["active_neurons"] == 1
    assert readout.diagnostics["samples_in_cumulative_statistics"] == 2
    assert readout.diagnostics["samples_in_leverage_statistics"] == 2
    assert 0.0 < readout.diagnostics["mean_normalized_leverage"] < 1.0


def test_recruited_neuron_accumulates_maturity_evidence() -> None:
    readout = CumulativeMaturityReadout(
        2, 2, max_neurons=2, min_center_distance=0.01
    )
    feature = np.array([1.0, 0.0])
    target = np.array([0.0, 1.0])
    for _ in range(8):
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)
    assert readout.diagnostics["active_neurons"] == 1
    assert readout.diagnostics["mean_neuron_maturity"] > 0.8
    assert readout.diagnostics["stored_raw_samples"] == 0


def test_maturity_variants_run_locked_with_bounded_capacity() -> None:
    rng = np.random.default_rng(8)
    labels = np.repeat(np.arange(2), 5)
    split = DigitsSplit(
        train_images=rng.uniform(size=(len(labels), 8, 8)),
        train_labels=labels,
        test_images=rng.uniform(size=(4, 8, 8)),
        test_labels=np.repeat(np.arange(2), 2),
    )
    for kind in (
        "maturity",
        "maturity_entropy",
        "maturity_leverage",
        "key_value",
        "key_value_entropy",
    ):
        result = run_digits_model(
            kind,
            "class_ordered",
            DigitsExperimentConfig(
                hidden_size=6,
                window=2,
                seed=8,
                maturity_max_neurons=4,
            ),
            split=split,
        )
        assert result["bounded_state"]
        assert result["weights_locked_during_evaluation"]
        assert result["maturity_diagnostics"][
            "samples_in_cumulative_statistics"
        ] == len(labels)


def test_maturity_checkpoint_round_trip(tmp_path) -> None:
    config = DigitsExperimentConfig(
        hidden_size=8, seed=5, maturity_max_neurons=4
    )
    learner = build_digits_learner("maturity_entropy", config)
    rng = np.random.default_rng(5)
    for label in (2, 7, 2, 1):
        _process_digit_image(
            learner, rng.uniform(size=(8, 8)), target=_digit_target(label)
        )
    path = save_checkpoint(learner, tmp_path / "maturity.npz")
    evaluation_image = rng.uniform(size=(8, 8))
    expected = _process_digit_image(learner, evaluation_image, target=None)

    restored = build_digits_learner("maturity_entropy", config)
    restore_checkpoint(restored, path)
    actual = _process_digit_image(restored, evaluation_image, target=None)
    np.testing.assert_allclose(actual, expected)
    assert restored.readout.diagnostics == learner.readout.diagnostics


def test_key_value_unit_learns_key_locality_and_value() -> None:
    readout = KeyValueMaturityReadout(
        2,
        2,
        max_neurons=1,
        rbf_width=0.05,
        minimum_key_variance=4e-4,
        maximum_key_variance=3.6e-3,
    )
    target = np.array([0.0, 1.0])
    origin = np.zeros(2)
    prediction = readout.predict(origin)
    readout.update(origin, target, prediction)

    initial_key = readout.neuron_centers[0].copy()
    nearby = np.array([0.02, 0.0])
    prediction = readout.predict(nearby)
    readout.update(nearby, target, prediction)

    assert readout.neuron_centers[0, 0] > initial_key[0]
    assert readout.key_weight[0] > readout.key_prior_strength
    assert np.all(readout.key_variance[0] >= readout.minimum_key_variance)
    assert np.all(readout.key_variance[0] <= readout.maximum_key_variance)
    assert np.linalg.norm(readout.expanded_weights[:, 2]) > 0.0
    assert readout.diagnostics["adaptive_keys"]


def test_key_maturity_reduces_center_movement() -> None:
    readout = KeyValueMaturityReadout(1, 2, max_neurons=1)
    target = np.array([0.0, 1.0])
    prediction = readout.predict(np.array([0.0]))
    readout.update(np.array([0.0]), target, prediction)

    observation = np.array([0.02])
    before = readout.neuron_centers.copy()
    prediction = readout.predict(observation)
    readout.update(observation, target, prediction)
    first_movement = float(np.linalg.norm(readout.neuron_centers - before))

    before = readout.neuron_centers.copy()
    prediction = readout.predict(observation)
    readout.update(observation, target, prediction)
    second_movement = float(np.linalg.norm(readout.neuron_centers - before))

    assert 0.0 < second_movement < first_movement


def test_key_value_checkpoint_round_trip(tmp_path) -> None:
    config = DigitsExperimentConfig(
        hidden_size=8, seed=11, maturity_max_neurons=4
    )
    learner = build_digits_learner("key_value_entropy", config)
    rng = np.random.default_rng(11)
    for label in (5, 1, 5, 3):
        _process_digit_image(
            learner, rng.uniform(size=(8, 8)), target=_digit_target(label)
        )
    path = save_checkpoint(learner, tmp_path / "key-value.npz")

    restored = build_digits_learner("key_value_entropy", config)
    restore_checkpoint(restored, path)
    np.testing.assert_allclose(
        restored.readout.key_weight, learner.readout.key_weight
    )
    np.testing.assert_allclose(restored.readout.key_m2, learner.readout.key_m2)
    np.testing.assert_allclose(
        restored.readout.key_variance, learner.readout.key_variance
    )
    np.testing.assert_allclose(
        restored.readout.neuron_centers, learner.readout.neuron_centers
    )

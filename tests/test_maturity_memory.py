import numpy as np

from no_backprop.checkpoint import restore_checkpoint, save_checkpoint
from no_backprop.digits import DigitsSplit
from no_backprop.experiment import (
    DigitsExperimentConfig,
    _digit_target,
    _process_digit_image,
    build_digits_learner,
    run_digits_model,
)
from no_backprop.readouts import CumulativeMaturityReadout, _normalized_entropy


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
    fields = CumulativeMaturityReadout.__dataclass_fields__
    assert "forgetting_factor" not in fields
    assert "decay" not in fields


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
    for kind in ("maturity", "maturity_entropy"):
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

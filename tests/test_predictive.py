import numpy as np

from no_backprop.checkpoint import restore_checkpoint, save_checkpoint
from no_backprop.digits import DigitsSplit
from no_backprop.experiment import (
    DigitsExperimentConfig,
    _digit_target,
    _evaluate_digits_locked,
    _process_digit_image,
    build_digits_learner,
    run_digits_model,
)
from no_backprop.frontend_comparison import (
    PredictiveRepresentationConfig,
    run_predictive_representation_comparison,
)
from no_backprop.predictive import OnlinePredictiveSpatialClassifier


def test_predictive_context_masks_each_target_quadrant() -> None:
    learner = build_digits_learner(
        "managed16_predictive_conv",
        DigitsExperimentConfig(hidden_size=64, seed=3),
    )
    assert isinstance(learner, OnlinePredictiveSpatialClassifier)
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    feature_map = learner.encoder.feature_map(image)
    contexts, targets = learner._context_target_pairs(image)

    assert contexts.shape == (4, 69)
    assert targets.shape == (4, 16)
    for index, (row, column) in enumerate(learner.block_starts):
        masked = contexts[index, :64].reshape(4, 4, 4)
        np.testing.assert_array_equal(
            masked[:, row : row + 2, column : column + 2], 0.0
        )
        expected = feature_map[:, row : row + 2, column : column + 2]
        np.testing.assert_array_equal(targets[index], expected.reshape(-1))
        np.testing.assert_array_equal(
            contexts[index, 64:68], np.eye(4)[index]
        )
        assert contexts[index, -1] == 1.0


def test_predictive_learning_occurs_only_after_preupdate_prediction() -> None:
    learner = build_digits_learner(
        "managed16_predictive_conv",
        DigitsExperimentConfig(hidden_size=64, seed=5),
    )
    assert isinstance(learner, OnlinePredictiveSpatialClassifier)
    image = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    predictor_before = learner.predictor.weights.copy()
    classifier_before = learner.readout.weights.copy()

    prediction = learner.predict(image)
    pending_representation = learner._pending_features[:-1].copy()
    np.testing.assert_array_equal(learner.predictor.weights, predictor_before)
    np.testing.assert_array_equal(learner.readout.weights, classifier_before)
    np.testing.assert_array_equal(pending_representation, 0.0)

    learner.learn(_digit_target(2))
    assert not np.array_equal(learner.predictor.weights, predictor_before)
    assert not np.array_equal(learner.readout.weights, classifier_before)
    assert prediction.shape == (10,)
    assert learner.diagnostics["predictor_images"] == 1
    assert learner.diagnostics["predictor_updates"] == 4


def test_predictive_rls_reduces_repeated_target_error_without_forgetting() -> None:
    learner = build_digits_learner(
        "managed16_predictive_conv",
        DigitsExperimentConfig(hidden_size=64, seed=7),
    )
    assert isinstance(learner, OnlinePredictiveSpatialClassifier)
    image = np.arange(64, dtype=np.float64).reshape(8, 8) / 63.0
    initial = learner.representation_diagnostics(np.stack((image,)))
    for _ in range(12):
        _process_digit_image(learner, image, target=_digit_target(1))
    final = learner.representation_diagnostics(np.stack((image,)))

    assert final["target_prediction_mse"] < initial["target_prediction_mse"]
    assert learner.predictor.forgetting_factor == 1.0
    assert learner.diagnostics["stored_raw_samples"] == 0
    assert learner.diagnostics["predictor_updates"] == 48


def test_predictive_evaluation_is_locked_and_reports_noncollapsed_features() -> None:
    learner = build_digits_learner(
        "managed16_predictive_conv",
        DigitsExperimentConfig(hidden_size=64, seed=11),
    )
    assert isinstance(learner, OnlinePredictiveSpatialClassifier)
    rng = np.random.default_rng(11)
    training = rng.uniform(size=(20, 8, 8))
    for index, image in enumerate(training):
        _process_digit_image(
            learner, image, target=_digit_target(index % 3)
        )
    before_updates = learner.diagnostics["predictor_updates"]
    evaluation = _evaluate_digits_locked(
        learner, training[:8], np.arange(8) % 3
    )

    assert evaluation["weights_unchanged"]
    assert evaluation["transient_state_restored"]
    assert learner.diagnostics["predictor_updates"] == before_updates
    representation = evaluation["representation_diagnostics"]
    assert representation["effective_rank"] > 1.0
    assert representation["mean_feature_variance"] > 0.0


def test_predictive_checkpoint_round_trip(tmp_path) -> None:
    config = DigitsExperimentConfig(hidden_size=64, seed=13)
    learner = build_digits_learner("managed16_predictive_conv", config)
    rng = np.random.default_rng(13)
    for label in (1, 5, 1, 9, 5):
        _process_digit_image(
            learner, rng.uniform(size=(8, 8)), target=_digit_target(label)
        )
    path = save_checkpoint(learner, tmp_path / "predictive.npz")
    evaluation_image = rng.uniform(size=(8, 8))
    expected = _process_digit_image(learner, evaluation_image, target=None)

    restored = build_digits_learner("managed16_predictive_conv", config)
    restore_checkpoint(restored, path)
    actual = _process_digit_image(restored, evaluation_image, target=None)
    np.testing.assert_allclose(actual, expected)
    assert restored.diagnostics == learner.diagnostics
    np.testing.assert_array_equal(
        restored.predictor.inverse_correlation,
        learner.predictor.inverse_correlation,
    )


def test_predictive_digits_model_preserves_online_invariants() -> None:
    rng = np.random.default_rng(17)
    labels = np.repeat(np.arange(2), 5)
    split = DigitsSplit(
        train_images=rng.uniform(size=(len(labels), 8, 8)),
        train_labels=labels,
        test_images=rng.uniform(size=(4, 8, 8)),
        test_labels=np.repeat(np.arange(2), 2),
    )
    result = run_digits_model(
        "managed16_predictive_conv",
        "shuffled",
        DigitsExperimentConfig(hidden_size=64, seed=17),
        split=split,
    )

    assert result["bounded_state"]
    assert result["weights_locked_during_evaluation"]
    assert result["readout_feature_width"] == 64
    assert result["predictive_diagnostics"]["predictor_images"] == len(labels)
    assert result["predictive_diagnostics"]["predictor_updates"] == 4 * len(labels)
    assert result["representation_diagnostics"]["feature_width"] == 64


def test_predictive_comparison_reports_factor_free_contract() -> None:
    result = run_predictive_representation_comparison(
        PredictiveRepresentationConfig(
            seeds=(19,),
            test_per_class=2,
            protocols=("shuffled",),
            include_drift=False,
        )
    )
    invariants = result["invariants"]
    assert invariants["same_downstream_memory"]
    assert invariants["same_readout_feature_width"]
    assert invariants["four_predictor_updates_per_training_image"]
    assert invariants["every_training_image_updates_predictor"]
    assert not invariants["predictor_uses_backpropagation"]
    assert invariants["predictor_forgetting_factor"] == 1.0
    assert invariants["predictor_stored_raw_samples"] == 0
    summary = result["summary"]["predictive_representation"]["shuffled"]
    assert summary["effective_rank"]["mean"] > 0.0
    learning = result["summary"]["predictive_learning"]["shuffled"]
    assert (
        learning["final_target_prediction_mse"]["mean"]
        < learning["initial_target_prediction_mse"]["mean"]
    )

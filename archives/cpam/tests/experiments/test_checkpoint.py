import numpy as np
import pytest

from continual_core.checkpoint import restore_checkpoint, save_checkpoint
from experiments.legacy import (
    DigitsExperimentConfig,
    SignalExperimentConfig,
    _digit_target,
    _process_digit_image,
    build_digits_learner,
    build_signal_learner,
)
from continual_core.streams import iter_nonstationary_signal


def test_checkpoint_round_trip_preserves_prediction(tmp_path) -> None:
    config = SignalExperimentConfig(steps=20, hidden_size=10, seed=4)
    learner = build_signal_learner("rls", config)
    events = list(iter_nonstationary_signal(12, regime_length=4, seed=4))
    for event in events[:10]:
        learner.predict(event.observation)
        learner.learn(event.target)
    path = save_checkpoint(learner, tmp_path / "learner.npz")
    expected = learner.predict(events[10].observation)
    learner.learn(events[10].target)

    restored = build_signal_learner("rls", config)
    restore_checkpoint(restored, path)
    actual = restored.predict(events[10].observation)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "kind", ["diagonal_rls", "block_rls", "prototype", "protected"]
)
def test_scalable_memory_checkpoint_round_trip(tmp_path, kind: str) -> None:
    config = DigitsExperimentConfig(hidden_size=8, block_size=3, seed=5)
    learner = build_digits_learner(kind, config)
    rng = np.random.default_rng(5)
    training_image = rng.uniform(size=(8, 8))
    evaluation_image = rng.uniform(size=(8, 8))
    _process_digit_image(learner, training_image, target=_digit_target(2))
    path = save_checkpoint(learner, tmp_path / f"{kind}.npz")
    expected = _process_digit_image(learner, evaluation_image, target=None)

    restored = build_digits_learner(kind, config)
    restore_checkpoint(restored, path)
    actual = _process_digit_image(restored, evaluation_image, target=None)
    np.testing.assert_allclose(actual, expected)

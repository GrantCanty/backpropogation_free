import numpy as np

from no_backprop.checkpoint import restore_checkpoint, save_checkpoint
from no_backprop.experiment import SignalExperimentConfig, build_signal_learner
from no_backprop.streams import iter_nonstationary_signal


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

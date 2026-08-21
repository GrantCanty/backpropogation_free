from no_backprop.experiment import SignalExperimentConfig, build_signal_learner
from no_backprop.streams import iter_nonstationary_signal


def test_core_state_size_does_not_grow_with_stream_length() -> None:
    config = SignalExperimentConfig(steps=2_000, hidden_size=16, regime_length=200)
    learner = build_signal_learner("lms", config)
    initial = learner.state_nbytes
    for event in iter_nonstationary_signal(
        config.steps, regime_length=config.regime_length, seed=config.seed
    ):
        learner.predict(event.observation)
        learner.learn(event.target)
    assert learner.state_nbytes == initial

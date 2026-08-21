from baselines.lms import LMSReadout
from baselines.reservoir import OnlineReservoir, ReservoirConfig
from continual_core.streams import iter_nonstationary_signal


def test_core_state_size_does_not_grow_with_stream_length() -> None:
    steps = 2_000
    regime_length = 200
    seed = 7
    learner = OnlineReservoir(
        ReservoirConfig(input_size=1, hidden_size=16, output_size=1, seed=seed),
        LMSReadout(17, 1, seed=seed),
    )
    initial = learner.state_nbytes
    for event in iter_nonstationary_signal(
        steps, regime_length=regime_length, seed=seed
    ):
        learner.predict(event.observation)
        learner.learn(event.target)
    assert learner.state_nbytes == initial

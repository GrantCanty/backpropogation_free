import numpy as np
import pytest

from continual_core.protocols import ProtocolError
from baselines.lms import LMSReadout
from baselines.reservoir import OnlineReservoir, ReservoirConfig


def build_learner() -> OnlineReservoir:
    config = ReservoirConfig(hidden_size=8, seed=4)
    return OnlineReservoir(config, LMSReadout(9, 1, learning_rate=0.1))


def test_predict_then_learn_order_is_enforced() -> None:
    learner = build_learner()
    with pytest.raises(ProtocolError):
        learner.learn(np.array([0.0]))
    learner.predict(np.array([0.25]))
    with pytest.raises(ProtocolError):
        learner.predict(np.array([0.5]))
    learner.learn(np.array([0.3]))
    learner.predict(np.array([0.5]))


def test_nan_target_advances_state_without_updating_readout() -> None:
    learner = build_learner()
    before = learner.readout.weights.copy()
    learner.predict(np.array([0.2]))
    learner.learn(np.array([np.nan]))
    np.testing.assert_array_equal(before, learner.readout.weights)

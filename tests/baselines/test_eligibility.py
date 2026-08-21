import numpy as np
import pytest

from baselines.eligibility import EligibilityConfig, EligibilityReservoir
from continual_core.protocols import ProtocolError
from baselines.lms import LMSReadout
from baselines.reservoir import ReservoirConfig


def build_plastic_learner() -> EligibilityReservoir:
    reservoir = ReservoirConfig(input_size=2, hidden_size=4, seed=2)
    readout = LMSReadout(5, 1, learning_rate=0.1)
    return EligibilityReservoir(
        reservoir,
        readout,
        EligibilityConfig(
            trace_decay=0.5,
            recurrent_learning_rate=0.01,
            input_learning_rate=0.01,
            seed=3,
        ),
    )


def test_trace_updates_locally_and_decays() -> None:
    learner = build_plastic_learner()
    learner.predict(np.array([1.0, -0.5]))
    first_input_trace = learner.input_eligibility.copy()
    learner.learn(np.array([np.nan]))
    assert np.linalg.norm(first_input_trace) > 0.0

    learner.predict(np.zeros(2))
    np.testing.assert_allclose(
        learner.input_eligibility,
        0.5 * first_input_trace,
        rtol=1e-12,
        atol=1e-12,
    )


def test_feedback_changes_recurrent_weights_without_history() -> None:
    learner = build_plastic_learner()
    learner.predict(np.array([1.0, 0.0]))
    learner.learn(np.array([np.nan]))
    learner.predict(np.array([0.0, 1.0]))
    before = learner.recurrent_weights.copy()
    learner.learn(np.array([1.0]))
    assert not np.array_equal(before, learner.recurrent_weights)
    assert learner.diagnostics["updates"] == 1


def test_plastic_protocol_still_enforces_order() -> None:
    learner = build_plastic_learner()
    with pytest.raises(ProtocolError):
        learner.learn(np.array([1.0]))

import numpy as np
import pytest

from continual_core.evaluation import (
    DirectUpdateAdapter,
    evaluate_classification_locked,
    train_classification_profiled,
    train_prequential,
)
from continual_core.protocols import ProtocolError
from continual_core.state import locked_state, state_nbytes


class CumulativeLinearLearner:
    def __init__(self) -> None:
        self.weights = np.zeros((2, 2), dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features

    def update(
        self,
        features: np.ndarray,
        target: np.ndarray,
        prediction: np.ndarray,
    ) -> None:
        self.weights += np.outer(target - prediction, features)

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights}

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {}


def _target(label: int) -> np.ndarray:
    target = np.zeros(2, dtype=np.float64)
    target[label] = 1.0
    return target


def test_shared_evaluator_runs_a_baseline_without_candidate_imports() -> None:
    learner = CumulativeLinearLearner()
    adapter = DirectUpdateAdapter()
    events = [
        (np.array([1.0, 0.0]), _target(0)),
        (np.array([0.0, 1.0]), _target(1)),
    ]
    training = train_prequential(learner, events, adapter)
    result = evaluate_classification_locked(
        learner,
        [event[0] for event in events],
        [0, 1],
        adapter,
        _target,
    )
    assert training["samples"] == 2
    assert result["accuracy"] == 1.0
    assert result["weights_unchanged"]
    assert result["state_bytes"] == state_nbytes(learner)


def test_locked_state_restores_transient_and_rejects_persistent_mutation() -> None:
    class State:
        def __init__(self) -> None:
            self.weight = np.array([1.0])
            self.activity = np.array([2.0])

        @property
        def persistent_state(self):
            return {"weight": self.weight}

        @property
        def transient_state(self):
            return {"activity": self.activity}

    state = State()
    with locked_state(state):
        state.activity[0] = 9.0
    assert state.activity[0] == 2.0
    with pytest.raises(ProtocolError, match="persistent state"):
        with locked_state(state):
            state.weight[0] = 3.0


def test_profiled_evaluator_owns_checkpoints_and_resource_metrics() -> None:
    learner = CumulativeLinearLearner()
    events = [
        (np.array([1.0, 0.0]), _target(0)),
        (np.array([0.0, 1.0]), _target(1)),
    ]
    result = train_classification_profiled(
        learner,
        [events[:1], events[1:]],
        DirectUpdateAdapter(),
        {"all": ([event[0] for event in events], [0, 1])},
        _target,
        sample_efficiency_steps=(1, 2),
    )
    assert result["samples"] == 2
    assert len(result["checkpoints"]) == 2
    assert result["checkpoints"][-1]["evaluation_sets"]["all"][
        "weights_unchanged"
    ]
    assert result["state_bytes_before"] == result["state_bytes_after"]
    assert result["nonfinite_state_values"] == 0

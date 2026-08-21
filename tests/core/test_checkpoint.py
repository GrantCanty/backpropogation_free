import numpy as np
import pytest

from continual_core.checkpoint import restore_checkpoint, save_checkpoint
from continual_core.protocols import ProtocolError


class StatefulFixture:
    def __init__(self) -> None:
        self.weights = np.arange(4, dtype=np.float64)
        self.activity = np.ones(2, dtype=np.float64)
        self._pending_prediction = None

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights}

    @property
    def transient_state(self) -> dict[str, np.ndarray]:
        return {"activity": self.activity}


def test_generic_checkpoint_round_trip(tmp_path) -> None:
    original = StatefulFixture()
    original.activity[:] = 3.0
    path = save_checkpoint(original, tmp_path / "state.npz")
    restored = StatefulFixture()
    restored.weights[:] = -1.0
    restored.activity[:] = -1.0
    restore_checkpoint(restored, path)
    np.testing.assert_array_equal(restored.weights, original.weights)
    np.testing.assert_array_equal(restored.activity, original.activity)


def test_checkpoint_rejects_open_prediction_transaction(tmp_path) -> None:
    learner = StatefulFixture()
    learner._pending_prediction = np.zeros(1)
    with pytest.raises(ProtocolError, match="between predict and update"):
        save_checkpoint(learner, tmp_path / "state.npz")

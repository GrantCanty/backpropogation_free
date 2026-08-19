import numpy as np
import pytest

from no_backprop.readouts import LMSReadout, RLSReadout


@pytest.mark.parametrize(
    "readout",
    [
        LMSReadout(3, 1, learning_rate=0.5),
        RLSReadout(3, 1, regularization=1.0),
    ],
)
def test_online_update_moves_prediction_toward_target(readout) -> None:
    features = np.array([1.0, -0.5, 1.0])
    target = np.array([0.75])
    before = readout.predict(features)
    readout.update(features, target, before)
    after = readout.predict(features)
    assert abs(target[0] - after[0]) < abs(target[0] - before[0])


def test_rls_tracks_all_auxiliary_memory() -> None:
    readout = RLSReadout(4, 2)
    assert readout.state_nbytes == readout.weights.nbytes + readout.inverse_correlation.nbytes

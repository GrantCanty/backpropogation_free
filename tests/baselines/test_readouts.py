import numpy as np
import pytest

from baselines.fast_slow import ProtectedFastSlowReadout
from baselines.lms import LMSReadout
from baselines.prototype import PrototypeReadout
from baselines.rls import BlockRLSReadout, DiagonalRLSReadout, RLSReadout


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


@pytest.mark.parametrize(
    "readout",
    [
        DiagonalRLSReadout(5, 2),
        BlockRLSReadout(5, 2, block_size=2),
    ],
)
def test_approximate_rls_moves_prediction_toward_target(readout) -> None:
    features = np.array([1.0, -0.5, 0.25, 0.1, 1.0])
    target = np.array([1.0, 0.0])
    before = readout.predict(features)
    readout.update(features, target, before)
    after = readout.predict(features)
    assert np.linalg.norm(target - after) < np.linalg.norm(target - before)


def test_rls_approximations_reduce_auxiliary_state() -> None:
    exact = RLSReadout(17, 3)
    diagonal = DiagonalRLSReadout(17, 3)
    blocked = BlockRLSReadout(17, 3, block_size=4)
    assert diagonal.state_nbytes < blocked.state_nbytes < exact.state_nbytes


def test_prototype_memory_retains_separately_observed_classes() -> None:
    readout = PrototypeReadout(3, 2)
    examples = (
        (np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0])),
        (np.array([0.0, 1.0, 1.0]), np.array([0.0, 1.0])),
    )
    for features, target in examples:
        readout.update(features, target, readout.predict(features))
    assert np.argmax(readout.predict(examples[0][0])) == 0
    assert np.argmax(readout.predict(examples[1][0])) == 1


def test_protected_memory_has_bounded_fast_and_slow_state() -> None:
    readout = ProtectedFastSlowReadout(4, 2)
    before = readout.state_nbytes
    features = np.array([1.0, -0.5, 0.25, 1.0])
    target = np.array([1.0, 0.0])
    readout.update(features, target, readout.predict(features))
    assert readout.state_nbytes == before
    assert readout.slow_memory.counts[0] == 1.0
    assert np.linalg.norm(readout.fast_weights) > 0.0

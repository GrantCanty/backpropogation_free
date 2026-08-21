import numpy as np
import pytest

from baselines.fast_slow import ProtectedFastSlowReadout
from baselines.lms import LMSReadout
from baselines.prototype import PrototypeReadout
from baselines.rls import BlockRLSReadout, DiagonalRLSReadout, RLSReadout
from baselines.rpls import RecursivePLSReadout


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


def test_one_block_rls_matches_exact_factor_one_rls() -> None:
    exact = RLSReadout(4, 2, regularization=0.7, forgetting_factor=1.0)
    blocked = BlockRLSReadout(
        4, 2, regularization=0.7, forgetting_factor=1.0, block_size=4
    )
    rng = np.random.default_rng(3)
    for _ in range(12):
        features = rng.normal(size=4)
        target = rng.normal(size=2)
        for readout in (exact, blocked):
            prediction = readout.predict(features)
            readout.update(features, target, prediction)
    assert np.allclose(blocked.weights, exact.weights, atol=1e-11)
    assert np.allclose(
        blocked.inverse_blocks[0], exact.inverse_correlation, atol=1e-11
    )


def test_full_component_rpls_matches_batch_ridge() -> None:
    rng = np.random.default_rng(11)
    features = rng.normal(size=(16, 4))
    targets = rng.normal(size=(16, 2))
    readout = RecursivePLSReadout(
        4, 2, components=4, regularization=0.8
    )
    for observation, target_value in zip(features, targets):
        prediction = readout.predict(observation)
        readout.update(observation, target_value, prediction)
    expected = np.linalg.solve(
        features.T @ features + 0.8 * np.eye(4), features.T @ targets
    ).T
    assert np.allclose(readout.weights, expected, atol=1e-9)


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

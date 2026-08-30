import numpy as np
import pytest

from methods.covariance_sketch import FrequentDirectionsRidgeReadout


def test_sketch_matches_batch_ridge_before_first_compression() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(5, 7))
    targets = rng.normal(size=(5, 2))
    readout = FrequentDirectionsRidgeReadout(
        7, 2, sketch_rank=3, regularization=0.6
    )
    for observation, target in zip(features, targets):
        prediction = readout.predict(observation)
        readout.update(observation, target, prediction)
    expected = np.linalg.solve(
        features.T @ features + 0.6 * np.eye(7), features.T @ targets
    ).T
    assert np.allclose(readout.weights, expected, atol=1e-10)


def test_sketch_state_is_bounded_and_finite_after_compressions() -> None:
    rng = np.random.default_rng(9)
    readout = FrequentDirectionsRidgeReadout(9, 3, sketch_rank=2)
    initial_bytes = readout.state_nbytes
    for _ in range(30):
        features = rng.normal(size=9)
        target = rng.normal(size=3)
        prediction = readout.predict(features)
        readout.update(features, target, prediction)
    assert readout.state_nbytes == initial_bytes
    assert readout.diagnostics["compressions"] > 0
    assert readout.diagnostics["bounded_state"] is True
    assert readout.diagnostics["replay"] is False
    assert all(np.all(np.isfinite(array)) for array in readout.persistent_arrays)


def test_sketch_rejects_nonfinite_observations() -> None:
    readout = FrequentDirectionsRidgeReadout(4, 2, sketch_rank=1)
    with pytest.raises(ValueError, match="finite"):
        readout.update(np.array([0.0, np.nan, 1.0, 2.0]), np.ones(2), np.zeros(2))


def test_sketch_rank_must_fit_feature_coordinates() -> None:
    with pytest.raises(ValueError, match="sketch_rank"):
        FrequentDirectionsRidgeReadout(4, 2, sketch_rank=4)

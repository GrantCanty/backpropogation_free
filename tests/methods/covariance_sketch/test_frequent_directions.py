import numpy as np

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
    assert all(np.all(np.isfinite(array)) for array in readout.persistent_arrays)

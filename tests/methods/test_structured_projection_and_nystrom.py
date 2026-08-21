import numpy as np

from baselines.os_elm import OSELMFeatureMap
from baselines.rls import RLSReadout
from methods.nystrom_memory import NystromCovarianceReadout
from methods.structured_projection import SparseSignedFeatureMap


def test_sparse_projection_is_deterministic_and_exactly_sparse() -> None:
    first = SparseSignedFeatureMap(12, 7, 3, seed=4)
    second = SparseSignedFeatureMap(12, 7, 3, seed=4)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.signs, second.signs)
    assert np.all([len(set(row.tolist())) == 3 for row in first.indices])
    assert first.state_nbytes == sum(array.nbytes for array in first.persistent_arrays)
    assert first.diagnostics["stored_dense_weights"] is False


def test_sparse_projection_matches_its_declared_output_contract() -> None:
    feature_map = SparseSignedFeatureMap(5, 4, 2, seed=8)
    output = feature_map.transform(np.arange(5.0))
    assert output.shape == (5,)
    assert output[-1] == 1.0
    assert OSELMFeatureMap(5, 4, seed=8).output_size == feature_map.output_size


def test_identity_probe_nystrom_matches_factor_one_rls() -> None:
    rng = np.random.default_rng(9)
    nystrom = NystromCovarianceReadout(5, 2, rank=5, regularization=0.7, seed=2)
    nystrom.probes[...] = np.eye(5)
    exact = RLSReadout(5, 2, regularization=0.7, forgetting_factor=1.0)
    for _ in range(12):
        features = rng.normal(size=5)
        target = rng.normal(size=2)
        nystrom.update(features, target, nystrom.predict(features))
        exact.update(features, target, exact.predict(features))
    assert np.allclose(nystrom.weights, exact.weights, atol=1e-10)
    assert nystrom.diagnostics["recruitment"] is False
    assert nystrom.diagnostics["forgetting_factor"] == 1.0

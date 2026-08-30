import numpy as np

from baselines.artmap import FuzzyARTMAPReadout
from baselines.kernel_rls import ALDKernelRLSReadout
from baselines.os_elm import OSELMFeatureMap, OnlineSequentialELMReadout
from baselines.ran import ResourceAllocatingNetworkReadout


def target(label: int, classes: int = 2) -> np.ndarray:
    result = np.zeros(classes, dtype=np.float64)
    result[label] = 1.0
    return result


def update(readout, features: np.ndarray, label: int) -> np.ndarray:
    prediction = readout.predict(features)
    readout.update(features, target(label, readout.output_size), prediction)
    return prediction


def test_os_elm_is_deterministic_bounded_and_learns() -> None:
    first = OnlineSequentialELMReadout(3, 2, hidden_size=8, seed=7)
    second = OnlineSequentialELMReadout(3, 2, hidden_size=8, seed=7)
    assert np.array_equal(first.hidden_weights, second.hidden_weights)
    state = first.state_nbytes
    features = np.array([0.2, -0.1, 1.0])
    before = first.predict(features)
    update(first, features, 1)
    after = first.predict(features)
    assert after[1] > before[1]
    assert first.state_nbytes == state


def test_public_os_elm_feature_map_matches_integrated_baseline() -> None:
    readout = OnlineSequentialELMReadout(3, 2, hidden_size=8, seed=7)
    feature_map = OSELMFeatureMap(3, 8, seed=7)
    observation = np.array([0.2, -0.1, 1.0])
    assert np.array_equal(
        feature_map.transform(observation), readout._features(observation)
    )


def test_integrated_os_elm_matches_public_features_plus_exact_rls() -> None:
    from baselines.rls import RLSReadout

    integrated = OnlineSequentialELMReadout(
        3, 2, hidden_size=5, seed=13, regularization=0.7
    )
    feature_map = OSELMFeatureMap(3, 5, seed=13)
    solver = RLSReadout(
        6, 2, regularization=0.7, forgetting_factor=1.0
    )
    rng = np.random.default_rng(17)
    for _ in range(10):
        observation = rng.normal(size=3)
        target_value = rng.normal(size=2)
        integrated_prediction = integrated.predict(observation)
        expanded = feature_map.transform(observation)
        solver_prediction = solver.predict(expanded)
        assert np.allclose(integrated_prediction, solver_prediction, atol=1e-12)
        integrated.update(observation, target_value, integrated_prediction)
        solver.update(expanded, target_value, solver_prediction)
    assert np.allclose(integrated.weights, solver.weights, atol=1e-11)
    assert np.allclose(
        integrated.inverse_correlation,
        solver.inverse_correlation,
        atol=1e-11,
    )


def test_ald_krls_expands_then_uses_reduced_updates_at_budget() -> None:
    readout = ALDKernelRLSReadout(
        2,
        2,
        max_dictionary_size=2,
        kernel_width=0.05,
        ald_tolerance=1e-4,
    )
    state = readout.state_nbytes
    update(readout, np.array([0.0, 0.0]), 0)
    update(readout, np.array([1.0, 1.0]), 1)
    centers = readout.dictionary.copy()
    update(readout, np.array([-1.0, -1.0]), 0)
    assert int(readout.active_count[0]) == 2
    assert int(readout.capacity_rejections[0]) == 1
    assert np.array_equal(readout.dictionary, centers)
    assert readout.state_nbytes == state
    assert np.all(np.isfinite(readout.coefficients))


def test_ran_recruits_fixed_centers_with_a_hard_capacity() -> None:
    readout = ResourceAllocatingNetworkReadout(
        2,
        2,
        max_neurons=2,
        error_threshold=0.5,
        distance_threshold=0.1,
    )
    update(readout, np.array([0.0, 0.0]), 0)
    update(readout, np.array([1.0, 1.0]), 1)
    centers = readout.centers.copy()
    update(readout, np.array([-1.0, -1.0]), 0)
    assert int(readout.active_count[0]) == 2
    assert np.array_equal(readout.centers, centers)
    assert int(readout.capacity_rejections[0]) >= 1


def test_fuzzy_artmap_match_tracking_separates_conflicting_labels() -> None:
    readout = FuzzyARTMAPReadout(
        2, 2, max_categories=3, vigilance=0.5, choice=0.001
    )
    features = np.array([0.2, -0.2])
    update(readout, features, 0)
    update(readout, features, 1)
    assert int(readout.active_count[0]) == 2
    assert int(readout.match_tracking_resets[0]) >= 1
    assert set(readout.category_labels[:2].astype(int)) == {0, 1}

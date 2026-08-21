import numpy as np
import pytest

import baselines.analytic_memory as analytic_memory
from baselines.analytic_memory import (
    AnalyticMemoryComparisonConfig,
    run_analytic_memory_comparison,
)
from baselines.analytic_readouts import (
    ALDKernelRLSReadout,
    FuzzyARTMAPReadout,
    OnlineSequentialELMReadout,
    ResourceAllocatingNetworkReadout,
)
from no_backprop.digits import DigitsSplit


def _target(label: int, classes: int = 2) -> np.ndarray:
    result = np.zeros(classes, dtype=np.float64)
    result[label] = 1.0
    return result


def _update(readout, features: np.ndarray, label: int) -> np.ndarray:
    prediction = readout.predict(features)
    readout.update(features, _target(label, readout.output_size), prediction)
    return prediction


def test_os_elm_is_deterministic_bounded_and_learns() -> None:
    first = OnlineSequentialELMReadout(3, 2, hidden_size=8, seed=7)
    second = OnlineSequentialELMReadout(3, 2, hidden_size=8, seed=7)
    assert np.array_equal(first.hidden_weights, second.hidden_weights)
    state = first.state_nbytes
    features = np.array([0.2, -0.1, 1.0])
    before = first.predict(features)
    _update(first, features, 1)
    after = first.predict(features)
    assert after[1] > before[1]
    assert first.state_nbytes == state


def test_ald_krls_expands_then_uses_reduced_updates_at_budget() -> None:
    readout = ALDKernelRLSReadout(
        2,
        2,
        max_dictionary_size=2,
        kernel_width=0.05,
        ald_tolerance=1e-4,
    )
    state = readout.state_nbytes
    _update(readout, np.array([0.0, 0.0]), 0)
    _update(readout, np.array([1.0, 1.0]), 1)
    centers = readout.dictionary.copy()
    _update(readout, np.array([-1.0, -1.0]), 0)
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
    _update(readout, np.array([0.0, 0.0]), 0)
    _update(readout, np.array([1.0, 1.0]), 1)
    centers = readout.centers.copy()
    _update(readout, np.array([-1.0, -1.0]), 0)
    assert int(readout.active_count[0]) == 2
    assert np.array_equal(readout.centers, centers)
    assert int(readout.capacity_rejections[0]) >= 1


def test_fuzzy_artmap_match_tracking_separates_conflicting_labels() -> None:
    readout = FuzzyARTMAPReadout(
        2, 2, max_categories=3, vigilance=0.5, choice=0.001
    )
    features = np.array([0.2, -0.2])
    _update(readout, features, 0)
    _update(readout, features, 1)
    assert int(readout.active_count[0]) == 2
    assert int(readout.match_tracking_resets[0]) >= 1
    assert set(readout.category_labels[:2].astype(int)) == {0, 1}


def _small_split(seed: int = 3) -> DigitsSplit:
    rng = np.random.default_rng(seed)
    train_labels = np.repeat(np.arange(2), 5)
    test_labels = np.repeat(np.arange(2), 2)
    return DigitsSplit(
        train_images=rng.uniform(size=(len(train_labels), 8, 8)),
        train_labels=train_labels,
        test_images=rng.uniform(size=(len(test_labels), 8, 8)),
        test_labels=test_labels,
    )


def test_comparison_tunes_off_test_and_enforces_state_ceilings(monkeypatch) -> None:
    split = _small_split(11)
    monkeypatch.setattr(analytic_memory, "load_digits_split", lambda **_: split)
    result = run_analytic_memory_comparison(
        AnalyticMemoryComparisonConfig(
            test_seeds=(7,),
            development_seeds=(2,),
            test_per_class=1,
            phase_domains=(
                "original",
                "inversion",
                "original",
                "inversion",
            ),
            krls_grid=((0.1, 0.01),),
            ran_grid=((0.1, 0.5, 0.1),),
            artmap_grid=((0.7, 0.001),),
            os_elm_grid=((0.5, 1.0),),
        )
    )
    assert result["development"]["seeds_disjoint_from_test"]
    assert set(result["selected_parameters"]) == {
        "ald_krls",
        "ran",
        "fuzzy_artmap",
        "os_elm",
    }
    assert len(result["models"]) == 11
    invariants = result["invariants"]
    assert invariants["same_fixed_signed_magnitude_frontend"]
    assert invariants["one_update_per_training_image"]
    assert invariants["weights_locked_during_evaluation"]
    assert invariants["bounded_preallocated_state"]
    assert invariants["finite_persistent_state"]
    assert invariants["analytic_baselines_do_not_exceed_named_cpam_budget"]
    for family in result["state_budget_allocations"].values():
        for allocation in family.values():
            assert allocation["actual_bytes"] <= allocation["target_bytes"]


def test_comparison_rejects_overlapping_development_and_test_seeds() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        AnalyticMemoryComparisonConfig(
            test_seeds=(3,), development_seeds=(3,)
        )

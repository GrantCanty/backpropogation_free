import numpy as np
import pytest

import experiments.cpam_capstone as memory_capstone
from continual_core.datasets.digits import DigitsSplit
from experiments.cpam_capstone import (
    MemoryCapstoneConfig,
    run_memory_capstone,
)
from baselines.rls import RLSReadout
from methods.cpam.readouts import ManagedProbationaryMaturityReadout


def _small_split(seed: int = 3) -> DigitsSplit:
    rng = np.random.default_rng(seed)
    train_labels = np.repeat(np.arange(2), 8)
    test_labels = np.repeat(np.arange(2), 3)
    return DigitsSplit(
        train_images=rng.uniform(size=(len(train_labels), 8, 8)),
        train_labels=train_labels,
        test_images=rng.uniform(size=(len(test_labels), 8, 8)),
        test_labels=test_labels,
    )


def test_capstone_builds_matched_factor_and_memory_models() -> None:
    config = MemoryCapstoneConfig(
        seeds=(3,),
        phase_domains=("original", "inversion", "original", "inversion"),
        mature_capacities=(4,),
        forgetting_factors=(1.0, 0.99),
    )
    cumulative = memory_capstone._build_learner("rls_ff_1", config, 3)
    discounted = memory_capstone._build_learner("rls_ff_0.99", config, 3)
    managed = memory_capstone._build_learner("managed_memory_4", config, 3)

    assert cumulative.encoder.name == discounted.encoder.name == managed.encoder.name
    assert isinstance(cumulative.readout, RLSReadout)
    assert cumulative.readout.forgetting_factor == 1.0
    assert discounted.readout.forgetting_factor == 0.99
    assert isinstance(managed.readout, ManagedProbationaryMaturityReadout)
    assert managed.readout.max_neurons == 4
    assert managed.readout.candidate_capacity == 16


def test_capstone_reports_returns_saturation_and_locked_evaluation(
    monkeypatch,
) -> None:
    split = _small_split(5)
    monkeypatch.setattr(memory_capstone, "load_digits_split", lambda **_: split)
    result = run_memory_capstone(
        MemoryCapstoneConfig(
            seeds=(5,),
            test_per_class=1,
            phase_domains=(
                "original",
                "inversion",
                "original",
                "inversion",
            ),
            mature_capacities=(2,),
            forgetting_factors=(1.0, 0.99),
        )
    )

    assert result["dataset"]["downloaded_data_bytes"] == 0
    assert result["dataset"]["labels_preserved_by_regimes"]
    assert result["baseline"] == "rls_ff_1"
    assert result["invariants"] == {
        "same_fixed_signed_magnitude_frontend": True,
        "one_label_update_per_training_image": True,
        "weights_locked_during_evaluation": True,
        "mature_centers_are_frozen": True,
        "bounded_state": True,
        "raw_samples_stored_by_models": 0,
        "rls_factor_one_discards_historical_weight": False,
    }
    assert set(result["summary"]["returns"]) == {"original", "inversion"}
    assert set(result["summary"]["phases"]) == {"1", "2", "3", "4"}
    factor = result["summary"]["factor_frontier"]["0.99"]
    assert factor["approximate_effective_history"] == pytest.approx(100.0)
    saturation = result["summary"]["capacity_frontier"]["2"]["saturation"]
    assert saturation["capacity_filled_by_end"]["mean"] == 1.0
    assert saturation["final_capacity_utilization"]["mean"] == 1.0
    managed_run = result["runs"][0]["models"]["managed_memory_2"]
    assert managed_run["capacity_filled_phase"] is not None
    assert all(
        phase["memory"]["maximum_existing_center_shift"] == 0.0
        for phase in managed_run["phases"]
    )


def test_capstone_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="original must recur"):
        MemoryCapstoneConfig(
            seeds=(3,),
            phase_domains=("original", "inversion", "inversion"),
        )
    with pytest.raises(ValueError, match="inversion must recur"):
        MemoryCapstoneConfig(
            seeds=(3,),
            phase_domains=("original", "original", "inversion"),
        )
    with pytest.raises(ValueError, match="include 1"):
        MemoryCapstoneConfig(seeds=(3,), forgetting_factors=(0.99,))
    with pytest.raises(ValueError, match="positive and unique"):
        MemoryCapstoneConfig(seeds=(3,), mature_capacities=(8, 8))

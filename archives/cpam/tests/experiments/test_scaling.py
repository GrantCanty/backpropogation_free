from experiments.scaling import (
    BlankImageScalingConfig,
    FeatureWidthScalingConfig,
    MemoryCapacityScalingConfig,
    _projected_managed_memory_bytes,
    run_blank_image_scaling,
    run_feature_width_scaling,
    run_memory_capacity_scaling,
)
from methods.cpam.readouts import ManagedProbationaryMaturityReadout


def test_blank_image_scaling_is_lazy_and_bounded() -> None:
    result = run_blank_image_scaling(
        BlankImageScalingConfig(
            sample_counts=(12,),
            image_sizes=(4,),
            hidden_size=6,
            classes=2,
            block_size=3,
            memory_neurons=3,
            memory_candidates=2,
            kinds=("rls", "diagonal_rls", "prototype", "managed_probation"),
        )
    )
    assert all(run["bounded_state"] for run in result["runs"])
    assert all(run["dataset_storage_bytes"] == 4 * 4 * 8 for run in result["runs"])
    assert all(run["downloaded_data_bytes"] == 0 for run in result["runs"])


def test_feature_width_scaling_reports_measured_and_projected_state() -> None:
    result = run_feature_width_scaling(
        FeatureWidthScalingConfig(
            feature_widths=(5, 9),
            projected_widths=(5, 9, 17),
            updates=12,
            classes=2,
            block_size=4,
            kinds=("rls", "diagonal_rls", "block_rls"),
        )
    )
    assert len(result["measured"]) == 6
    assert len(result["projected"]) == 9
    exact = {
        item["feature_width"]: item["projected_state_bytes"]
        for item in result["projected"]
        if item["model"] == "rls"
    }
    diagonal = {
        item["feature_width"]: item["projected_state_bytes"]
        for item in result["projected"]
        if item["model"] == "diagonal_rls"
    }
    assert exact[17] > diagonal[17]


def test_managed_memory_projection_matches_allocated_arrays() -> None:
    readout = ManagedProbationaryMaturityReadout(
        5, 2, max_neurons=3, max_candidates=2
    )
    assert _projected_managed_memory_bytes(5, 2, 3, 2) == readout.state_nbytes


def test_memory_capacity_scaling_is_bounded_and_separates_axes() -> None:
    result = run_memory_capacity_scaling(
        MemoryCapacityScalingConfig(
            feature_width=5,
            key_capacities=(1, 3),
            candidate_capacities=(1, 2, 4),
            fixed_key_capacity=3,
            fixed_candidate_capacity=2,
            updates=12,
            classes=2,
            projected_widths=(5, 9),
            projected_key_capacities=(3, 7),
        )
    )
    assert len(result["key_capacity_runs"]) == 2
    assert len(result["candidate_capacity_runs"]) == 3
    assert len(result["projected"]) == 4
    assert result["downloaded_data_bytes"] == 0
    assert all(run["bounded_state"] for run in result["key_capacity_runs"])
    assert all(run["bounded_state"] for run in result["candidate_capacity_runs"])
    assert [run["active_keys"] for run in result["key_capacity_runs"]] == [1, 3]
    assert all(run["prefilled_keys"] for run in result["key_capacity_runs"])
    assert not any(
        run["prefilled_keys"] for run in result["candidate_capacity_runs"]
    )
    key_state = [run["state_bytes_before"] for run in result["key_capacity_runs"]]
    candidate_state = [
        run["state_bytes_before"] for run in result["candidate_capacity_runs"]
    ]
    assert key_state[1] > key_state[0]
    assert candidate_state[2] > candidate_state[1] > candidate_state[0]

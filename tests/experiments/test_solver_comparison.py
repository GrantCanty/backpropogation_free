import json

import numpy as np

from experiments.solver_comparison import (
    SolverComparisonConfig,
    run_fixed_solver_study,
    run_solver_comparison,
    write_solver_artifacts,
)


def tiny_config() -> SolverComparisonConfig:
    return SolverComparisonConfig(
        hidden_size=8,
        test_per_class=2,
        augmentation_copies=0,
        development_seeds=(3,),
        heldout_seeds=(7,),
        development_events_per_segment=1,
        heldout_events_per_segment=2,
        regularization_grid=(1.0,),
        lms_learning_rates=(0.1,),
        block_sizes=(2,),
        rpls_components=(2,),
        sketch_ranks=(2,),
    )


def test_solver_comparison_runs_matched_disjoint_protocol(tmp_path) -> None:
    result = run_solver_comparison(tiny_config())
    assert result["protocol"]["backpropagation"] is False
    assert result["protocol"]["development_seeds_disjoint"] is True
    expected = {
        "lms",
        "exact_rls",
        "diagonal_rls",
        "block_rls",
        "rpls",
        "fd_ridge",
        "memory_matched_exact_rls",
    }
    assert set(result["summary"]) == expected
    assert len(result["heldout_runs"]) == 2 * len(expected)
    fingerprints: dict[str, set[str]] = {}
    for run in result["heldout_runs"]:
        assert run["training"]["samples"] == 20
        assert run["numerical_stability"]["nonfinite_state_values"] == 0
        assert all(
            checkpoint["evaluation_sets"]["all"]["weights_unchanged"]
            for checkpoint in run["training"]["checkpoints"]
        )
        fingerprints.setdefault(run["protocol"], set()).add(
            run["matched_problem_sha256"]
        )
    assert all(len(values) == 1 for values in fingerprints.values())

    destination = write_solver_artifacts(result, tmp_path / "artifacts")
    raw = list((destination / "raw").glob("*.json"))
    assert len(raw) == len(result["heldout_runs"])
    assert json.loads((destination / "comparison.json").read_text())[
        "experiment"
    ] == "os_elm_solver_comparison"
    assert (destination / "REPORT.md").is_file()


def test_development_and_heldout_seed_overlap_is_rejected() -> None:
    try:
        SolverComparisonConfig(
            development_seeds=(1, 2), heldout_seeds=(2, 3)
        )
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping seeds were accepted")


def test_fixed_study_derives_input_and_class_sizes_from_npz(tmp_path) -> None:
    rng = np.random.default_rng(4)
    path = tmp_path / "two_class.npz"
    np.savez(
        path,
        train_images=rng.random((20, 2, 3)),
        train_labels=np.asarray([0, 1] * 10),
        test_images=rng.random((8, 2, 3)),
        test_labels=np.asarray([0, 1] * 4),
    )
    config = SolverComparisonConfig(
        dataset="npz",
        dataset_path=str(path),
        hidden_size=4,
        test_per_class=2,
        augmentation_copies=0,
        development_seeds=(3,),
        heldout_seeds=(7,),
        development_events_per_segment=1,
        heldout_events_per_segment=1,
        regularization_grid=(1.0,),
        lms_learning_rates=(0.1,),
        block_sizes=(2,),
        rpls_components=(2,),
        sketch_ranks=(1,),
    )
    result = run_fixed_solver_study(
        config,
        {
            "exact_rls": {"regularization": 1.0},
            "fd_ridge": {"regularization": 1.0, "sketch_rank": 1},
        },
    )
    assert len(result["heldout_runs"]) == 6
    assert {
        run["assumptions"]["full_feature_map_state_bytes"]
        for run in result["heldout_runs"]
    } == {4 * 6 * 8 + 4 * 8}

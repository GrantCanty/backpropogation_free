from scripts.run_nystrom_rank_sweep import build_parser
from scripts.run_frequent_directions_sweep import (
    build_parser as build_fd_parser,
    select_development_candidate,
)
from scripts.run_projection_memory_study import parser as build_ablation_parser
from experiments.projection_memory_study import ProjectionMemoryConfig, run_projection_memory_study
from methods.covariance_sketch import FrequentDirectionsRidgeReadout
import numpy as np


def test_rank_sweep_defaults_to_large_width_development_grid() -> None:
    args = build_parser().parse_args([])

    assert str(args.output) == "results/nystrom_rank_sweep"
    assert args.widths == (1024,)
    assert args.ranks == (64, 96, 128, 192, 256)
    assert args.regularizations == (0.1, 1.0, 10.0)
    assert args.seeds == (100, 101, 102, 103, 104)
    assert args.sparse_fan_in == 8


def test_rank_sweep_accepts_explicit_comma_separated_grid() -> None:
    args = build_parser().parse_args(
        ["--widths", "512,1024", "--ranks", "64,128", "--regularizations", "0.1,1"]
    )

    assert args.widths == (512, 1024)
    assert args.ranks == (64, 128)
    assert args.regularizations == (0.1, 1.0)


def test_final_ablation_accepts_independent_solver_regularization() -> None:
    args = build_ablation_parser().parse_args(
        ["--exact-regularization", "0.5", "--nystrom-regularization", "10"]
    )

    assert args.exact_regularization == 0.5
    assert args.nystrom_regularization == 10.0


def test_frequent_directions_defaults_and_selection_rule() -> None:
    args = build_fd_parser().parse_args([])
    assert args.ranks == (64, 128, 192, 256, 320, 384)
    assert args.development_seeds == (100, 101)
    assert args.confirmation_seeds == tuple(range(102, 120))
    studies = [
        {"candidates": [
            {"condition": "sparse_fd__rank_128__ridge_1", "parameters": {"feature": "sparse", "rank": 128, "regularization": 1.0}, "mean_final_locked_accuracy": 0.70, "total_memory_reduction_vs_sparse_exact": 0.10},
            {"condition": "dense_fd__rank_128__ridge_1", "parameters": {"feature": "dense", "rank": 128, "regularization": 1.0}, "mean_final_locked_accuracy": 0.75, "total_memory_reduction_vs_sparse_exact": -0.10},
        ]},
        {"candidates": [
            {"condition": "sparse_fd__rank_128__ridge_1", "parameters": {"feature": "sparse", "rank": 128, "regularization": 1.0}, "mean_final_locked_accuracy": 0.72, "total_memory_reduction_vs_sparse_exact": 0.05},
            {"condition": "dense_fd__rank_128__ridge_1", "parameters": {"feature": "dense", "rank": 128, "regularization": 1.0}, "mean_final_locked_accuracy": 0.76, "total_memory_reduction_vs_sparse_exact": -0.05},
        ]},
    ]
    assert select_development_candidate(studies)["selected"]["condition"] == "sparse_fd__rank_128__ridge_1"


def test_projection_runner_accepts_injected_frequent_directions_builder(tmp_path) -> None:
    def builder(width, output_size, seed, rank, regularization):
        assert rank is not None
        return FrequentDirectionsRidgeReadout(width, output_size, rank, regularization, seed)

    config = ProjectionMemoryConfig(
        input_size=4, hidden_size=4, fan_ins=(2,), ranks=(2,),
        development_seeds=(3,)
    )
    segments = [[(np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([1.0, 0.0]))]]
    observation = np.asarray([0.4, 0.3, 0.2, 0.1])
    evaluation = {
        "all": ([observation, observation], [0, 1]),
        "class_0": ([observation], [0]),
        "class_1": ([observation], [1]),
    }
    result = run_projection_memory_study(
        config=config,
        seeds=(3,),
        conditions={"fd": {"feature_kind": "dense", "readout_kind": "frequent_directions", "rank": 2}},
        segments_by_seed={3: segments},
        evaluation_by_seed={3: evaluation},
        output=tmp_path,
        readout_builders={"frequent_directions": builder},
    )
    assert result["runs"][0]["diagnostics"]["algorithm"] == "frequent_directions_ridge"
    assert (tmp_path / "raw" / "fd__seed_3.json").is_file()

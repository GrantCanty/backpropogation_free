#!/usr/bin/env python3
"""Run dense- and sparse-projection Nyström rank/regularization candidates.

The sweep reuses existing dense- and sparse-exact raw artifacts for comparisons
and does not rerun those expensive references. Results are exploratory
development evidence; select here, then confirm once on fresh paired seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from continual_core.results import write_json_result
from experiments.projection_memory_study import (
    ProjectionMemoryConfig,
    materialize_projection_problem,
    run_projection_memory_study,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated values")
    return values


def _absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _condition_name(feature: str, rank: int, regularization: float) -> str:
    ridge = format(regularization, "g").replace(".", "p")
    return f"{feature}_nystrom__rank_{rank}__ridge_{ridge}"


def _final_accuracy(run: Mapping[str, Any]) -> float:
    return float(
        run["training"]["checkpoints"][-1]["evaluation_sets"]["all"]["accuracy"]
    )


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(9_173)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def _reference_runs(
    reference: Path, width: int, protocol: str, seeds: tuple[int, ...]
) -> dict[str, dict[int, dict[str, Any]]]:
    raw = reference / f"width_{width}" / protocol / "raw"
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for method in ("dense_exact", "sparse_exact"):
        result[method] = {}
        for seed in seeds:
            path = raw / f"{method}__seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing paired {method} reference {path}; use seeds already "
                    "present in --reference-results or run a paired reference first"
                )
            result[method][seed] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _comparison_summary(
    runs: list[dict[str, Any]],
    references: Mapping[str, Mapping[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["condition"]), []).append(run)
    summary: list[dict[str, Any]] = []
    for condition, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: int(item["seed"]))
        dense_exact = [
            references["dense_exact"][int(run["seed"])] for run in candidates
        ]
        sparse_exact = [
            references["sparse_exact"][int(run["seed"])] for run in candidates
        ]
        parameters = dict(candidates[0]["parameters"])
        matched_name = f"{parameters['feature']}_exact"
        matched_exact = dense_exact if matched_name == "dense_exact" else sparse_exact
        differences_dense = np.asarray(
            [
                _final_accuracy(run) - _final_accuracy(ref)
                for run, ref in zip(candidates, dense_exact)
            ],
            dtype=np.float64,
        )
        differences_sparse = np.asarray(
            [
                _final_accuracy(run) - _final_accuracy(ref)
                for run, ref in zip(candidates, sparse_exact)
            ],
            dtype=np.float64,
        )
        differences_matched = np.asarray(
            [
                _final_accuracy(run) - _final_accuracy(ref)
                for run, ref in zip(candidates, matched_exact)
            ],
            dtype=np.float64,
        )
        dense_lower, dense_upper = _bootstrap_interval(differences_dense)
        sparse_lower, sparse_upper = _bootstrap_interval(differences_sparse)
        matched_lower, matched_upper = _bootstrap_interval(differences_matched)
        candidate_solver = float(np.mean([run["resources"]["solver_bytes"] for run in candidates]))
        baseline_solver = float(np.mean([run["resources"]["solver_bytes"] for run in matched_exact]))
        candidate_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in candidates]))
        dense_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in dense_exact]))
        sparse_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in sparse_exact]))
        matched_total = dense_total if matched_name == "dense_exact" else sparse_total
        candidate_throughput = float(np.median([run["resources"]["throughput_events_per_second"] for run in candidates]))
        dense_throughput = float(np.median([run["resources"]["throughput_events_per_second"] for run in dense_exact]))
        sparse_throughput = float(np.median([run["resources"]["throughput_events_per_second"] for run in sparse_exact]))
        matched_throughput = (
            dense_throughput if matched_name == "dense_exact" else sparse_throughput
        )
        solver_reduction = 1.0 - candidate_solver / baseline_solver
        total_reduction_vs_dense = 1.0 - candidate_total / dense_total
        total_reduction_vs_sparse = 1.0 - candidate_total / sparse_total
        quality_vs_dense = dense_lower > -0.01
        quality_vs_sparse = sparse_lower > -0.01
        quality_vs_matched = matched_lower > -0.01
        latency_vs_dense = candidate_throughput >= dense_throughput
        latency_vs_sparse = candidate_throughput >= sparse_throughput
        latency_vs_matched = candidate_throughput >= matched_throughput
        total_reduction_vs_matched = 1.0 - candidate_total / matched_total
        summary.append(
            {
                "condition": condition,
                "parameters": parameters,
                "matched_exact_reference": matched_name,
                "seeds": [int(run["seed"]) for run in candidates],
                "mean_final_locked_accuracy": float(np.mean([_final_accuracy(run) for run in candidates])),
                "mean_paired_accuracy_delta_vs_dense_exact": float(np.mean(differences_dense)),
                "paired_accuracy_delta_vs_dense_exact_bootstrap_95_ci": [dense_lower, dense_upper],
                "mean_paired_accuracy_delta_vs_sparse_exact": float(np.mean(differences_sparse)),
                "paired_accuracy_delta_vs_sparse_exact_bootstrap_95_ci": [sparse_lower, sparse_upper],
                "mean_paired_accuracy_delta_vs_matched_exact": float(np.mean(differences_matched)),
                "paired_accuracy_delta_vs_matched_exact_bootstrap_95_ci": [matched_lower, matched_upper],
                "solver_memory_reduction": solver_reduction,
                "total_memory_reduction_vs_dense_exact": total_reduction_vs_dense,
                "total_memory_reduction_vs_sparse_exact": total_reduction_vs_sparse,
                "total_memory_reduction_vs_matched_exact": total_reduction_vs_matched,
                "median_throughput_ratio_vs_dense_exact": candidate_throughput / dense_throughput,
                "median_throughput_ratio_vs_sparse_exact": candidate_throughput / sparse_throughput,
                "median_throughput_ratio_vs_matched_exact": candidate_throughput / matched_throughput,
                "quality_gate_vs_dense_exact_pass": quality_vs_dense,
                "quality_gate_vs_sparse_exact_pass": quality_vs_sparse,
                "latency_gate_vs_dense_exact_pass": latency_vs_dense,
                "latency_gate_vs_sparse_exact_pass": latency_vs_sparse,
                "quality_gate_vs_matched_exact_pass": quality_vs_matched,
                "latency_gate_vs_matched_exact_pass": latency_vs_matched,
                "original_solver_gate_pass": solver_reduction >= 0.50,
                "original_total_gate_pass": total_reduction_vs_matched >= 0.25,
                "combined_total_50_percent_gate_pass": total_reduction_vs_dense >= 0.50,
                "all_original_gates_pass": bool(
                    quality_vs_matched and latency_vs_matched
                    and solver_reduction >= 0.50
                    and total_reduction_vs_matched >= 0.25
                ),
                "all_combined_tradeoff_gates_pass": bool(
                    quality_vs_dense and latency_vs_dense
                    and total_reduction_vs_dense >= 0.50
                ),
            }
        )
    return summary


def _write_report(summary: Mapping[str, Any], destination: Path) -> None:
    lines = ["# Dense and Sparse Nyström Rank Sweep", "",
             "Exploratory development sweep; confirmation requires fresh paired seeds.", ""]
    for study in summary["studies"]:
        lines.extend([f"## Width {study['width']} — {study['protocol']}", ""])
        for item in study["candidates"]:
            rank = item["parameters"]["rank"]
            ridge = item["parameters"]["regularization"]
            feature = item["parameters"]["feature"]
            matched = item["matched_exact_reference"].replace("_", "+")
            sparse_ci = item[
                "paired_accuracy_delta_vs_sparse_exact_bootstrap_95_ci"
            ]
            dense_ci = item[
                "paired_accuracy_delta_vs_dense_exact_bootstrap_95_ci"
            ]
            lines.append(
                f"- {feature}, rank {rank}, ridge {ridge:g}: vs {matched} accuracy "
                f"{item['mean_paired_accuracy_delta_vs_matched_exact']:+.4f} "
                f"(95% CI {item['paired_accuracy_delta_vs_matched_exact_bootstrap_95_ci'][0]:+.4f}, "
                f"{item['paired_accuracy_delta_vs_matched_exact_bootstrap_95_ci'][1]:+.4f}), "
                f"vs dense+exact {item['mean_paired_accuracy_delta_vs_dense_exact']:+.4f} "
                f"(95% CI {dense_ci[0]:+.4f}, {dense_ci[1]:+.4f}); solver memory "
                f"reduction {100 * item['solver_memory_reduction']:+.1f}%; "
                f"total reduction vs matched exact "
                f"{100 * item['total_memory_reduction_vs_matched_exact']:+.1f}%, "
                f"vs dense+exact {100 * item['total_memory_reduction_vs_dense_exact']:+.1f}%; "
                f"throughput vs matched exact "
                f"{item['median_throughput_ratio_vs_matched_exact']:.2f}×, "
                f"vs dense+exact {item['median_throughput_ratio_vs_dense_exact']:.2f}×; "
                f"original gates "
                f"{'PASS' if item['all_original_gates_pass'] else 'FAIL'}; "
                f"combined tradeoff "
                f"{'PASS' if item['all_combined_tradeoff_gates_pass'] else 'FAIL'}"
            )
        lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _absolute(args.output)
    reference = _absolute(args.reference_results)
    print(f"writing sweep artifacts to {output}", flush=True)
    print(f"reusing dense-exact and sparse-exact references from {reference}", flush=True)
    widths = (16,) if args.smoke else args.widths
    ranks = (4, 8) if args.smoke else args.ranks
    regularizations = (1.0,) if args.smoke else args.regularizations
    seeds = (args.smoke_seed,) if args.smoke else args.seeds
    events = 4 if args.smoke else args.train_events_per_segment
    tests = 4 if args.smoke else args.test_per_class
    studies: list[dict[str, Any]] = []
    for width in widths:
        for protocol in ("shuffled_augmented", "class_ordered"):
            segments_by_seed: dict[int, Any] = {}
            evaluation_by_seed: dict[int, Any] = {}
            input_size: int | None = None
            for seed in seeds:
                segments, evaluation, current_input = materialize_projection_problem(
                    dataset=args.dataset,
                    seed=seed,
                    protocol=protocol,
                    events_per_segment=events,
                    test_per_class=tests,
                    dataset_path=args.dataset_path,
                    allow_download=args.allow_download,
                    dataset_cache=args.dataset_cache,
                    augmentation_copies=args.augmentation_copies,
                    augmentation_max_shift=args.augmentation_max_shift,
                    augmentation_noise_std=args.augmentation_noise_std,
                )
                input_size = current_input if input_size is None else input_size
                if current_input != input_size:
                    raise RuntimeError("input size changed between paired seeds")
                segments_by_seed[seed] = segments
                evaluation_by_seed[seed] = evaluation
            assert input_size is not None
            config = ProjectionMemoryConfig(
                input_size=input_size,
                hidden_size=width,
                fan_ins=(args.sparse_fan_in,),
                ranks=tuple(ranks),
                nyström_regularizations=tuple(regularizations),
                development_seeds=tuple(seeds),
                confirmatory_seeds=tuple(),
            )
            conditions = {
                _condition_name(feature, rank, ridge): {
                    "feature_kind": feature,
                    "readout_kind": "nystrom",
                    "fan_in": (
                        min(args.sparse_fan_in, input_size)
                        if feature == "sparse"
                        else None
                    ),
                    "rank": min(rank, width + 1),
                    "regularization": ridge,
                }
                for feature in ("dense", "sparse")
                for rank in ranks
                for ridge in regularizations
            }
            destination = output / f"width_{width}" / protocol
            result = run_projection_memory_study(
                config=config,
                seeds=seeds,
                conditions=conditions,
                segments_by_seed=segments_by_seed,
                evaluation_by_seed=evaluation_by_seed,
                output=destination,
                resume=not args.no_resume,
                progress=lambda message, w=width, p=protocol: print(
                    f"width={w} protocol={p} {message}", flush=True
                ),
            )
            references = _reference_runs(reference, width, protocol, tuple(seeds))
            studies.append(
                {
                    "width": width,
                    "protocol": protocol,
                    "candidates": _comparison_summary(result["runs"], references),
                    "artifact_directory": str(destination),
                }
            )
    summary = {
        "schema_version": 1,
        "experiment": "nystrom_rank_sweep",
        "phase": "exploratory_development_after_rank32_result",
        "dataset": args.dataset,
        "configuration": {
            "widths": list(widths), "ranks": list(ranks),
            "regularizations": list(regularizations), "seeds": list(seeds),
            "sparse_fan_in": args.sparse_fan_in,
        },
        "reference_results": str(reference),
        "gate_note": (
            "Solver-only gates compare each Nyström candidate with exact RLS "
            "under the same feature map. The combined-system tradeoff also "
            "compares every candidate with dense exact RLS."
        ),
        "studies": studies,
    }
    write_json_result(summary, output / "rank_sweep.json")
    _write_report(summary, output / "RANK_SWEEP_REPORT.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "fashion_mnist", "npz"), default="digits")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/sparse_nystrom_rank_sweep"))
    parser.add_argument("--reference-results", type=Path, default=Path("results/projection_memory"))
    parser.add_argument("--widths", type=_ints, default=(1024,))
    parser.add_argument("--ranks", type=_ints, default=(64, 96, 128, 192, 256))
    parser.add_argument("--regularizations", type=_floats, default=(0.1, 1.0, 10.0))
    parser.add_argument("--seeds", type=_ints, default=(100, 101, 102, 103, 104))
    parser.add_argument("--train-events-per-segment", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--augmentation-max-shift", type=int, default=1)
    parser.add_argument("--augmentation-noise-std", type=float, default=0.03)
    parser.add_argument("--sparse-fan-in", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "npz" and args.dataset_path is None:
        raise SystemExit("--dataset-path is required for --dataset npz")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

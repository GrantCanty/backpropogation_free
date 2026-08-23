#!/usr/bin/env python3
"""Run only sparse-projection Nyström rank/regularization candidates.

The sweep reuses existing dense-exact raw artifacts for comparisons and does
not rerun that expensive reference.  Results are exploratory development
evidence; select here, then confirm once on fresh paired seeds.
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


def _condition_name(rank: int, regularization: float) -> str:
    ridge = format(regularization, "g").replace(".", "p")
    return f"sparse_nystrom__rank_{rank}__ridge_{ridge}"


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


def _reference_runs(reference: Path, width: int, protocol: str,
                    seeds: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    raw = reference / f"width_{width}" / protocol / "raw"
    result: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = raw / f"dense_exact__seed_{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing paired dense-exact reference {path}; use seeds already "
                "present in --reference-results or run a paired reference first"
            )
        result[seed] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _comparison_summary(runs: list[dict[str, Any]], references: Mapping[int, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["condition"]), []).append(run)
    summary: list[dict[str, Any]] = []
    for condition, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: int(item["seed"]))
        baseline = [references[int(run["seed"])] for run in candidates]
        differences = np.asarray(
            [_final_accuracy(run) - _final_accuracy(ref) for run, ref in zip(candidates, baseline)],
            dtype=np.float64,
        )
        lower, upper = _bootstrap_interval(differences)
        candidate_solver = float(np.mean([run["resources"]["solver_bytes"] for run in candidates]))
        baseline_solver = float(np.mean([run["resources"]["solver_bytes"] for run in baseline]))
        candidate_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in candidates]))
        baseline_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in baseline]))
        candidate_throughput = float(np.median([run["resources"]["throughput_events_per_second"] for run in candidates]))
        baseline_throughput = float(np.median([run["resources"]["throughput_events_per_second"] for run in baseline]))
        solver_reduction = 1.0 - candidate_solver / baseline_solver
        total_reduction = 1.0 - candidate_total / baseline_total
        quality_pass = lower > -0.01
        latency_pass = candidate_throughput >= baseline_throughput
        parameters = dict(candidates[0]["parameters"])
        summary.append(
            {
                "condition": condition,
                "parameters": parameters,
                "seeds": [int(run["seed"]) for run in candidates],
                "mean_final_locked_accuracy": float(np.mean([_final_accuracy(run) for run in candidates])),
                "mean_paired_accuracy_delta": float(np.mean(differences)),
                "paired_accuracy_delta_bootstrap_95_ci": [lower, upper],
                "solver_memory_reduction": solver_reduction,
                "total_memory_reduction": total_reduction,
                "median_throughput_ratio": candidate_throughput / baseline_throughput,
                "quality_gate_pass": quality_pass,
                "latency_gate_pass": latency_pass,
                "original_solver_gate_pass": solver_reduction >= 0.50,
                "original_total_gate_pass": total_reduction >= 0.25,
                "revised_total_50_percent_gate_pass": total_reduction >= 0.50,
                "all_original_gates_pass": bool(
                    quality_pass and latency_pass and solver_reduction >= 0.50
                    and total_reduction >= 0.25
                ),
                "all_revised_total_tradeoff_gates_pass": bool(
                    quality_pass and latency_pass and total_reduction >= 0.50
                ),
            }
        )
    return summary


def _write_report(summary: Mapping[str, Any], destination: Path) -> None:
    lines = ["# Sparse Nyström Rank Sweep", "",
             "Exploratory development sweep; confirmation requires fresh paired seeds.", ""]
    for study in summary["studies"]:
        lines.extend([f"## Width {study['width']} — {study['protocol']}", ""])
        for item in study["candidates"]:
            rank = item["parameters"]["rank"]
            ridge = item["parameters"]["regularization"]
            ci = item["paired_accuracy_delta_bootstrap_95_ci"]
            lines.append(
                f"- rank {rank}, ridge {ridge:g}: accuracy delta "
                f"{item['mean_paired_accuracy_delta']:+.4f} "
                f"(95% CI {ci[0]:+.4f}, {ci[1]:+.4f}); solver memory "
                f"reduction {100 * item['solver_memory_reduction']:+.1f}%; "
                f"total memory reduction {100 * item['total_memory_reduction']:+.1f}%; throughput "
                f"{item['median_throughput_ratio']:.2f}×; original gates "
                f"{'PASS' if item['all_original_gates_pass'] else 'FAIL'}; "
                f"revised total tradeoff "
                f"{'PASS' if item['all_revised_total_tradeoff_gates_pass'] else 'FAIL'}"
            )
        lines.append("")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = _absolute(args.output)
    reference = _absolute(args.reference_results)
    print(f"writing sweep artifacts to {output}", flush=True)
    print(f"reusing dense-exact references from {reference}", flush=True)
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
                _condition_name(rank, ridge): {
                    "feature_kind": "sparse",
                    "readout_kind": "nystrom",
                    "fan_in": min(args.sparse_fan_in, input_size),
                    "rank": min(rank, width + 1),
                    "regularization": ridge,
                }
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
        "experiment": "sparse_nystrom_rank_sweep",
        "phase": "exploratory_development_after_rank32_result",
        "dataset": args.dataset,
        "configuration": {
            "widths": list(widths), "ranks": list(ranks),
            "regularizations": list(regularizations), "seeds": list(seeds),
            "sparse_fan_in": args.sparse_fan_in,
        },
        "reference_results": str(reference),
        "gate_note": (
            "Original solver and total-memory gates are retained separately. "
            "The revised total-memory tradeoff is exploratory and does not "
            "retroactively change the original gate."
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

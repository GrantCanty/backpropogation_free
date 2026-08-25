#!/usr/bin/env python3
"""Run a single-width Frequent-Directions development or confirmation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from continual_core.results import write_json_result
from experiments.projection_memory_study import (
    ProjectionMemoryConfig,
    materialize_projection_problem,
    run_projection_memory_study,
)
from methods.covariance_sketch import FrequentDirectionsRidgeReadout


def _ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated values")
    return result


def _fd_builder(
    width: int, output_size: int, seed: int, rank: int | None, regularization: float
) -> FrequentDirectionsRidgeReadout:
    if rank is None:
        raise ValueError("Frequent-Directions readouts require a sketch rank")
    return FrequentDirectionsRidgeReadout(
        width, output_size, sketch_rank=rank, regularization=regularization, seed=seed
    )


def _absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _final_accuracy(run: Mapping[str, Any]) -> float:
    return float(run["training"]["checkpoints"][-1]["evaluation_sets"]["all"]["accuracy"])


def _references(
    root: Path, width: int, protocol: str, seeds: tuple[int, ...]
) -> dict[str, dict[int, dict[str, Any]]]:
    raw = root / f"width_{width}" / protocol / "raw"
    result: dict[str, dict[int, dict[str, Any]]] = {"dense_exact": {}, "sparse_exact": {}}
    for name in result:
        for seed in seeds:
            path = raw / f"{name}__seed_{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing paired reference artifact: {path}")
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("condition") != name or int(item.get("seed", -1)) != seed:
                raise ValueError(f"invalid paired reference artifact: {path}")
            result[name][seed] = item
    return result


def _condition(feature: str, rank: int, ridge: float) -> str:
    return f"{feature}_fd__rank_{rank}__ridge_{format(ridge, 'g').replace('.', 'p')}"


def _summarize(
    runs: list[dict[str, Any]], references: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["condition"]), []).append(run)
    result: list[dict[str, Any]] = []
    for name, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: int(item["seed"]))
        feature = str(candidates[0]["parameters"]["feature"])
        matched_name = f"{feature}_exact"
        matched = [references[matched_name][int(run["seed"])] for run in candidates]
        dense = [references["dense_exact"][int(run["seed"])] for run in candidates]
        sparse = [references["sparse_exact"][int(run["seed"])] for run in candidates]
        total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in candidates]))
        sparse_total = float(np.mean([run["resources"]["total_persistent_bytes"] for run in sparse]))
        result.append({
            "condition": name,
            "parameters": dict(candidates[0]["parameters"]),
            "seeds": [int(run["seed"]) for run in candidates],
            "mean_final_locked_accuracy": float(np.mean([_final_accuracy(run) for run in candidates])),
            "mean_paired_accuracy_delta_vs_matched_exact": float(np.mean([_final_accuracy(run) - _final_accuracy(ref) for run, ref in zip(candidates, matched)])),
            "mean_paired_accuracy_delta_vs_dense_exact": float(np.mean([_final_accuracy(run) - _final_accuracy(ref) for run, ref in zip(candidates, dense)])),
            "mean_paired_accuracy_delta_vs_sparse_exact": float(np.mean([_final_accuracy(run) - _final_accuracy(ref) for run, ref in zip(candidates, sparse)])),
            "mean_total_persistent_bytes": total,
            "mean_sparse_exact_total_persistent_bytes": sparse_total,
            "total_memory_reduction_vs_sparse_exact": 1.0 - total / sparse_total,
            "mean_peak_transient_training_bytes": float(np.mean([run["resources"]["peak_transient_training_bytes"] for run in candidates])),
            "median_throughput_events_per_second": float(np.median([run["resources"]["throughput_events_per_second"] for run in candidates])),
        })
    return result


def select_development_candidate(studies: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {}
    for study in studies:
        for candidate in study["candidates"]:
            records.setdefault(candidate["condition"], []).append(candidate)
    eligible = [
        values for values in records.values()
        if len(values) == 2 and all(item["total_memory_reduction_vs_sparse_exact"] > 0 for item in values)
    ]
    if not eligible:
        return {"selected": None, "reason": "no candidate was smaller than sparse exact in both protocols"}
    winner = max(
        eligible,
        key=lambda values: float(np.mean([item["mean_final_locked_accuracy"] for item in values])),
    )
    return {
        "selected": {
            "condition": winner[0]["condition"],
            "parameters": winner[0]["parameters"],
            "mean_final_locked_accuracy": float(np.mean([item["mean_final_locked_accuracy"] for item in winner])),
        },
        "rule": "highest mean final locked accuracy among candidates strictly smaller than sparse exact in both protocols",
    }


def _report(document: Mapping[str, Any]) -> str:
    lines = [f"# Frequent-Directions Width-{document['configuration']['width']} Study", ""]
    lines.append(f"Phase: {document['phase']}")
    lines.append("")
    for study in document["studies"]:
        lines.extend([f"## {study['protocol']}", ""])
        for item in study["candidates"]:
            lines.append(
                f"- {item['condition']}: final={item['mean_final_locked_accuracy']:.4f}; "
                f"vs sparse exact={item['mean_paired_accuracy_delta_vs_sparse_exact']:+.4f}; "
                f"state={item['mean_total_persistent_bytes']:.0f} bytes "
                f"({100 * item['total_memory_reduction_vs_sparse_exact']:+.1f}% vs sparse exact)."
            )
        lines.append("")
    selection = document.get("selection")
    if selection:
        lines.extend(["## Selection", "", json.dumps(selection, sort_keys=True), ""])
    return "\n".join(lines)


def _run(args: argparse.Namespace, *, phase: str, selected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    seeds = args.development_seeds if phase == "development" else args.confirmation_seeds
    if set(args.development_seeds) & set(args.confirmation_seeds):
        raise ValueError("development and confirmation seeds must be disjoint")
    if phase == "confirmation" and selected is None:
        raise ValueError("confirmation requires a selected development configuration")
    output = _absolute(args.output) / f"width_{args.width}" / phase
    studies: list[dict[str, Any]] = []
    if selected is None:
        parameter_sets = [
            (feature, rank, ridge)
            for feature in ("dense", "sparse")
            for rank in args.ranks
            for ridge in args.regularizations
        ]
    else:
        parameters = selected["parameters"]
        parameter_sets = [(str(parameters["feature"]), int(parameters["rank"]), float(parameters["regularization"]))]
    for protocol in ("shuffled_augmented", "class_ordered"):
        segments_by_seed: dict[int, Any] = {}
        evaluation_by_seed: dict[int, Any] = {}
        input_size: int | None = None
        for seed in seeds:
            segments, evaluation, current_input = materialize_projection_problem(
                dataset=args.dataset, seed=seed, protocol=protocol,
                events_per_segment=args.train_events_per_segment, test_per_class=args.test_per_class,
                dataset_path=args.dataset_path,
                allow_download=args.allow_download,
                dataset_cache=args.dataset_cache,
                augmentation_copies=args.augmentation_copies,
                augmentation_max_shift=args.augmentation_max_shift,
                augmentation_noise_std=args.augmentation_noise_std,
            )
            input_size = current_input if input_size is None else input_size
            segments_by_seed[seed] = segments
            evaluation_by_seed[seed] = evaluation
        assert input_size is not None
        config = ProjectionMemoryConfig(input_size=input_size, hidden_size=args.width,
            fan_ins=(args.sparse_fan_in,), ranks=args.ranks,
            nyström_regularizations=args.regularizations,
            development_seeds=args.development_seeds, confirmatory_seeds=args.confirmation_seeds)
        conditions = {
            _condition(feature, rank, ridge): {
                "feature_kind": feature, "readout_kind": "frequent_directions",
                "fan_in": args.sparse_fan_in if feature == "sparse" else None,
                "rank": rank, "regularization": ridge,
            }
            for feature, rank, ridge in parameter_sets
        }
        result = run_projection_memory_study(
            config=config, seeds=seeds, conditions=conditions,
            segments_by_seed=segments_by_seed, evaluation_by_seed=evaluation_by_seed,
            output=output / protocol, resume=not args.no_resume, protocol=protocol,
            timing_session_id=args.timing_session_id,
            rotate_condition_order=not args.fixed_condition_order,
            readout_builders={"frequent_directions": _fd_builder},
        )
        studies.append({"protocol": protocol, "artifact_directory": str(output / protocol),
                        "candidates": _summarize(
                            result["runs"],
                            _references(_absolute(args.reference_results), args.width, protocol, seeds),
                        )})
    document: dict[str, Any] = {
        "schema_version": 1, "experiment": "frequent_directions_rank_sweep",
        "dataset": args.dataset,
        "download_allowed": args.allow_download,
        "dataset_cache": str(_absolute(args.dataset_cache)) if args.dataset_cache else None,
        "phase": phase, "configuration": {"width": args.width, "ranks": list(args.ranks), "regularizations": list(args.regularizations),
        "seeds": list(seeds), "sparse_fan_in": args.sparse_fan_in},
        "reference_results": str(_absolute(args.reference_results)), "studies": studies,
    }
    if phase == "development":
        document["selection"] = select_development_candidate(studies)
    else:
        document["selected_from_development"] = dict(selected or {})
    write_json_result(document, output / "study.json")
    (output / "REPORT.md").write_text(_report(document), encoding="utf-8")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--output", type=Path, default=Path("results/frequent_directions_sweep"))
    parser.add_argument("--reference-results", type=Path, default=Path("results/projection_memory"))
    parser.add_argument("--development-artifact", type=Path)
    parser.add_argument("--dataset", choices=("digits", "fashion_mnist", "npz"), default="digits")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--ranks", type=_ints, default=(64, 128, 192, 256, 320, 384))
    parser.add_argument("--regularizations", type=_floats, default=(0.1, 1.0, 10.0))
    parser.add_argument("--development-seeds", type=_ints, default=(100, 101))
    parser.add_argument("--confirmation-seeds", type=_ints, default=tuple(range(102, 120)))
    parser.add_argument("--sparse-fan-in", type=int, default=8)
    parser.add_argument("--train-events-per-segment", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--augmentation-max-shift", type=int, default=1)
    parser.add_argument("--augmentation-noise-std", type=float, default=0.03)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fixed-condition-order", action="store_true")
    parser.add_argument("--timing-session-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.width <= 0:
        raise SystemExit("--width must be positive")
    if args.dataset == "npz" and args.dataset_path is None:
        raise SystemExit("--dataset-path is required for --dataset npz")
    selected = None
    if args.phase == "confirmation":
        artifact = _absolute(
            args.development_artifact
            or (args.output / f"width_{args.width}" / "development" / "study.json")
        )
        development = json.loads(artifact.read_text(encoding="utf-8"))
        selected = development.get("selection", {}).get("selected")
        if selected is None:
            raise SystemExit("development artifact has no eligible selected configuration")
    _run(args, phase=args.phase, selected=selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

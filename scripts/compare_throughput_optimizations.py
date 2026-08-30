#!/usr/bin/env python3
"""Run back-to-back benchmark comparing original vs optimized FD and Nyström across independent ranks."""

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
from methods.covariance_sketch import (
    FrequentDirectionsRidgeReadout,
    OriginalFrequentDirectionsRidgeReadout,
)
from methods.nystrom_memory import (
    NystromCovarianceReadout,
    OriginalNystromCovarianceReadout,
)


def _ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def _absolute(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _final_accuracy(run: Mapping[str, Any]) -> float:
    return float(run["training"]["checkpoints"][-1]["evaluation_sets"]["all"]["accuracy"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/throughput_comparison"))
    parser.add_argument("--dataset", choices=("digits", "fashion_mnist", "npz"), default="fashion_mnist")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--fd-ranks",
        "--fd-rank",
        dest="fd_ranks",
        type=_ints,
        default=(192,),
        help="One or more ranks to benchmark for Frequent Directions (e.g. 128,192,256)",
    )
    parser.add_argument(
        "--nystrom-ranks",
        "--nystrom-rank",
        dest="nystrom_ranks",
        type=_ints,
        default=(128,),
        help="One or more ranks to benchmark for Nyström (e.g. 64,128,192)",
    )
    parser.add_argument(
        "--implementation",
        choices=("both", "optimized", "original"),
        default="both",
        help="Benchmark 'both' (default) to compare, or selectively benchmark 'optimized' or 'original'.",
    )
    parser.add_argument("--fd-regularization", type=float, default=10.0)
    parser.add_argument("--nystrom-regularization", type=float, default=1.0)
    parser.add_argument("--sparse-fan-in", type=int, default=8)
    parser.add_argument("--seeds", type=_ints, default=(100, 101))
    parser.add_argument("--train-events-per-segment", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--protocols", type=str, default="shuffled_augmented,class_ordered")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timing-session-id")
    return parser


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    fd_str = "_".join(str(r) for r in args.fd_ranks)
    nys_str = "_".join(str(r) for r in args.nystrom_ranks)
    impl_suffix = f"_{args.implementation}" if args.implementation != "both" else ""
    output = _absolute(args.output) / f"width_{args.width}_fdranks_{fd_str}_nysranks_{nys_str}{impl_suffix}"
    output.mkdir(parents=True, exist_ok=True)
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    seeds = args.seeds

    print(f"=== Running Throughput Comparison Benchmark ===")
    print(f"Dataset: {args.dataset} | Width: {args.width} | Implementation: {args.implementation}")
    print(f"FD Ranks: {args.fd_ranks} (ridge={args.fd_regularization}) | Nyström Ranks: {args.nystrom_ranks} (ridge={args.nystrom_regularization})")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output}")

    readout_builders = {
        "frequent_directions_optimized": lambda w, o, s, r, reg: FrequentDirectionsRidgeReadout(
            w, o, sketch_rank=r, regularization=reg, seed=s, optimized=True
        ),
        "frequent_directions_original": lambda w, o, s, r, reg: OriginalFrequentDirectionsRidgeReadout(
            w, o, sketch_rank=r, regularization=reg, seed=s
        ),
        "nystrom_optimized": lambda w, o, s, r, reg: NystromCovarianceReadout(
            w, o, rank=r, seed=s, regularization=reg, optimized=True
        ),
        "nystrom_original": lambda w, o, s, r, reg: OriginalNystromCovarianceReadout(
            w, o, rank=r, seed=s, regularization=reg
        ),
    }

    conditions = {
        "sparse_exact": {
            "feature_kind": "sparse",
            "readout_kind": "exact",
            "fan_in": args.sparse_fan_in,
            "rank": None,
            "regularization": 1.0,
        }
    }

    include_orig = args.implementation in ("both", "original")
    include_opt = args.implementation in ("both", "optimized")

    for r in args.fd_ranks:
        if include_orig:
            conditions[f"original_sparse_fd__rank_{r}"] = {
                "feature_kind": "sparse",
                "readout_kind": "frequent_directions_original",
                "fan_in": args.sparse_fan_in,
                "rank": r,
                "regularization": args.fd_regularization,
            }
        if include_opt:
            conditions[f"optimized_sparse_fd__rank_{r}"] = {
                "feature_kind": "sparse",
                "readout_kind": "frequent_directions_optimized",
                "fan_in": args.sparse_fan_in,
                "rank": r,
                "regularization": args.fd_regularization,
            }

    for r in args.nystrom_ranks:
        if include_orig:
            conditions[f"original_sparse_nystrom__rank_{r}"] = {
                "feature_kind": "sparse",
                "readout_kind": "nystrom_original",
                "fan_in": args.sparse_fan_in,
                "rank": r,
                "regularization": args.nystrom_regularization,
            }
        if include_opt:
            conditions[f"optimized_sparse_nystrom__rank_{r}"] = {
                "feature_kind": "sparse",
                "readout_kind": "nystrom_optimized",
                "fan_in": args.sparse_fan_in,
                "rank": r,
                "regularization": args.nystrom_regularization,
            }

    all_ranks = tuple(set(args.fd_ranks + args.nystrom_ranks))
    all_results = {}

    for protocol in protocols:
        print(f"\n--- Protocol: {protocol} ---")
        segments_by_seed = {}
        evaluation_by_seed = {}
        input_size = None
        for seed in seeds:
            segments, evaluation, current_input = materialize_projection_problem(
                dataset=args.dataset,
                seed=seed,
                protocol=protocol,
                events_per_segment=args.train_events_per_segment,
                test_per_class=args.test_per_class,
                dataset_path=args.dataset_path,
                allow_download=args.allow_download,
                dataset_cache=args.dataset_cache,
            )
            input_size = current_input if input_size is None else input_size
            segments_by_seed[seed] = segments
            evaluation_by_seed[seed] = evaluation

        assert input_size is not None
        config = ProjectionMemoryConfig(
            input_size=input_size,
            hidden_size=args.width,
            fan_ins=(args.sparse_fan_in,),
            ranks=all_ranks,
            nyström_regularizations=(args.nystrom_regularization, args.fd_regularization),
            development_seeds=tuple(seeds),
            confirmatory_seeds=tuple(),
        )

        dest = output / protocol
        result = run_projection_memory_study(
            config=config,
            seeds=seeds,
            conditions=conditions,
            segments_by_seed=segments_by_seed,
            evaluation_by_seed=evaluation_by_seed,
            output=dest,
            resume=not args.no_resume,
            protocol=protocol,
            progress=lambda msg, p=protocol: print(f"[{p}] {msg}", flush=True),
            timing_session_id=args.timing_session_id,
            rotate_condition_order=False,
            readout_builders=readout_builders,
        )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in result["runs"]:
            grouped.setdefault(str(run["condition"]), []).append(run)

        rows = []
        ordered_cond_names = list(conditions.keys())
        for name in ordered_cond_names:
            if name in grouped:
                cruns = grouped[name]
                facc = float(np.mean([_final_accuracy(r) for r in cruns]))
                oacc = float(np.mean([r["training"]["online_accuracy"] for r in cruns]))
                smem = float(np.mean([r["resources"]["solver_bytes"] for r in cruns])) / 1024.0
                pmem = float(np.mean([r["resources"]["peak_transient_training_bytes"] for r in cruns])) / (1024.0 * 1024.0)
                tput = float(np.mean([r["resources"]["throughput_events_per_second"] for r in cruns]))
                lat = float(np.mean([r["resources"]["solver_update_latency"]["median_microseconds"] for r in cruns]))
                rows.append({
                    "condition": name,
                    "final_accuracy": facc,
                    "online_accuracy": oacc,
                    "solver_bytes_kb": smem,
                    "peak_transient_mb": pmem,
                    "throughput_events_per_sec": tput,
                    "median_update_latency_us": lat,
                })

        all_results[protocol] = rows

        print(f"\nBenchmark Summary: {protocol} (Width {args.width})")
        header = f"{'Condition':<38} | {'Final Acc':<10} | {'Online Acc':<10} | {'Solver Mem':<12} | {'Peak RAM':<10} | {'Throughput':<12} | {'Latency':<10}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['condition']:<38} | "
                f"{r['final_accuracy']*100:6.2f}%    | "
                f"{r['online_accuracy']*100:6.2f}%    | "
                f"{r['solver_bytes_kb']:8.1f} KB  | "
                f"{r['peak_transient_mb']:6.2f} MB | "
                f"{r['throughput_events_per_sec']:8.1f} e/s | "
                f"{r['median_update_latency_us']:6.1f} µs"
            )

    doc = {
        "schema_version": 1,
        "experiment": "throughput_optimizations_comparison",
        "dataset": args.dataset,
        "width": args.width,
        "fd_ranks": list(args.fd_ranks),
        "nystrom_ranks": list(args.nystrom_ranks),
        "fd_regularization": args.fd_regularization,
        "nystrom_regularization": args.nystrom_regularization,
        "sparse_fan_in": args.sparse_fan_in,
        "seeds": list(seeds),
        "protocols": all_results,
    }
    write_json_result(doc, output / "comparison_summary.json")
    print(f"\nSaved summary to {output / 'comparison_summary.json'}")
    return doc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

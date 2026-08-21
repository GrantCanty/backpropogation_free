#!/usr/bin/env python3
"""Run the four-condition frozen-projection/Nyström campaign.

Examples:
    PYTHONPATH=src python scripts/run_projection_memory_study.py --smoke
    PYTHONPATH=src python scripts/run_projection_memory_study.py \
        --dataset fashion_mnist --allow-download --output results/projection_memory
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from continual_core.datasets.classification import augment_image_split, load_classification_split
from continual_core.datasets.digits import build_digits_segments
from continual_core.results import write_json_result
from experiments.projection_memory_study import (
    ProjectionMemoryConfig,
    run_projection_memory_study,
)


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _build_problem(args: argparse.Namespace, seed: int, protocol: str,
                   width: int, events_per_segment: int,
                   test_per_class: int) -> tuple[ProjectionMemoryConfig, dict[int, Any], dict[int, Any]]:
    split = load_classification_split(
        args.dataset,
        test_per_class=test_per_class,
        seed=seed,
        dataset_path=args.dataset_path,
        allow_download=args.allow_download,
        cache_directory=args.dataset_cache,
    )
    split = augment_image_split(
        split,
        copies=args.augmentation_copies,
        max_shift=args.augmentation_max_shift,
        noise_std=args.augmentation_noise_std,
        seed=seed + 10_000,
    )
    stream = build_digits_segments(
        split.train_labels,
        protocol=protocol,  # type: ignore[arg-type]
        seed=seed + 20_000,
    )
    selected: list[np.ndarray] = []
    for segment in stream:
        selected.append(segment.indices[:events_per_segment])
    selected_indices = np.unique(np.concatenate(selected))
    observations = split.train_images.reshape(len(split.train_images), -1)
    targets = split.train_labels.astype(int)
    segments = {
        seed: [
            [
                (observations[int(index)], np.eye(len(np.unique(targets)))[targets[int(index)]])
                for index in indices
            ]
            for indices in selected
        ]
    }
    test_observations = split.test_images.reshape(len(split.test_images), -1)
    evaluation: dict[str, Any] = {
        "all": (list(test_observations), split.test_labels.astype(int).tolist())
    }
    for label in np.unique(split.test_labels):
        indices = np.flatnonzero(split.test_labels == label)
        evaluation[f"class_{int(label)}"] = (
            [test_observations[int(index)] for index in indices],
            [int(label)] * len(indices),
        )
    config = ProjectionMemoryConfig(
        input_size=int(observations.shape[1]),
        hidden_size=width,
        fan_ins=tuple(args.fan_ins),
        ranks=tuple(args.ranks),
        development_seeds=(seed,),
        confirmatory_seeds=tuple(),
    )
    return config, segments, {seed: evaluation}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        widths = (16,)
        seeds = (args.seed,)
        events = 4
        test_per_class = 4
        fan_in = min(4, 64)
        rank = 4
    else:
        widths = args.widths
        seeds = args.seeds
        events = args.train_events_per_segment
        test_per_class = args.test_per_class
        fan_in = args.sparse_fan_in
        rank = args.nystrom_rank

    output = Path(args.output)
    all_results: list[dict[str, Any]] = []
    for width in widths:
        for protocol in args.protocols:
            # The four conditions use the same raw event materialization.
            config, segments_by_seed, evaluation_by_seed = _build_problem(
                args, seeds[0], protocol, width, events, test_per_class
            )
            # Rebuild each seed's data independently, preserving paired ordering.
            for seed in seeds[1:]:
                _, segments, evaluation = _build_problem(
                    args, seed, protocol, width, events, test_per_class
                )
                segments_by_seed.update(segments)
                evaluation_by_seed.update(evaluation)
            conditions = {
                "dense_exact": {"feature_kind": "dense", "readout_kind": "exact"},
                "sparse_exact": {"feature_kind": "sparse", "readout_kind": "exact", "fan_in": min(fan_in, config.input_size)},
                "dense_nystrom": {"feature_kind": "dense", "readout_kind": "nystrom", "rank": rank},
                "sparse_nystrom": {"feature_kind": "sparse", "readout_kind": "nystrom", "fan_in": min(fan_in, config.input_size), "rank": rank},
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
            )
            all_results.append(result)
            print(f"completed width={width} protocol={protocol} seeds={len(seeds)}", flush=True)
    manifest = {"experiment": "structured_projection_nystrom_memory_campaign",
                "dataset": args.dataset, "results": [str(output / f"width_{w}" / p) for w in widths for p in args.protocols]}
    write_json_result(manifest, output / "campaign.json")
    return {"manifest": manifest, "studies": all_results}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("digits", "fashion_mnist", "npz"), default="digits")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/projection_memory"))
    parser.add_argument("--smoke", action="store_true", help="small local digits run")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=_ints, default=tuple(range(100, 120)))
    parser.add_argument("--widths", type=_ints, default=(64, 128, 256))
    parser.add_argument("--protocols", type=_ints, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-events-per-segment", type=int, default=1000)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--augmentation-max-shift", type=int, default=1)
    parser.add_argument("--augmentation-noise-std", type=float, default=0.03)
    parser.add_argument("--sparse-fan-in", type=int, default=8)
    parser.add_argument("--nystrom-rank", type=int, default=32)
    parser.add_argument("--fan-ins", type=_ints, default=(4, 8, 16, 32))
    parser.add_argument("--ranks", type=_ints, default=(8, 16, 32, 64))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.protocols = ("shuffled_augmented", "class_ordered")
    if args.dataset == "npz" and args.dataset_path is None:
        raise SystemExit("--dataset-path is required for --dataset npz")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

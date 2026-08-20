"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
import json

from no_backprop.experiment import (
    ContinualExperimentConfig,
    DelayedExperimentConfig,
    DigitsExperimentConfig,
    SignalExperimentConfig,
    run_continual_experiment,
    run_delayed_experiment,
    run_digits_experiment,
    run_signal_experiment,
    write_json_result,
)
from no_backprop.replication import ReplicationConfig, run_replication
from no_backprop.plotting import load_result, plot_result
from no_backprop.milestone6 import Milestone6Config, run_milestone6
from no_backprop.scaling import (
    BlankImageScalingConfig,
    FeatureWidthScalingConfig,
    run_scaling_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="no-backprop",
        description="Run bounded-memory online-learning experiments.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")
    signal = subparsers.add_parser("signal", help="run nonstationary signal benchmark")
    signal.add_argument("--steps", type=int, default=3_000)
    signal.add_argument("--regime-length", type=int, default=750)
    signal.add_argument("--hidden-size", type=int, default=64)
    signal.add_argument("--seed", type=int, default=7)
    signal.add_argument("--output", type=str)
    delayed = subparsers.add_parser("delayed", help="run delayed association benchmark")
    delayed.add_argument("--episodes", type=int, default=1_500)
    delayed.add_argument("--delay", type=int, default=8)
    delayed.add_argument("--hidden-size", type=int, default=48)
    delayed.add_argument("--seed", type=int, default=13)
    delayed.add_argument("--output", type=str)
    continual = subparsers.add_parser(
        "continual", help="run recurring-context classification benchmark"
    )
    continual.add_argument("--steps", type=int, default=4_000)
    continual.add_argument("--context-length", type=int, default=1_000)
    continual.add_argument("--hidden-size", type=int, default=48)
    continual.add_argument("--seed", type=int, default=17)
    continual.add_argument("--output", type=str)
    digits = subparsers.add_parser(
        "digits", help="run bundled 8x8 image benchmarks without a download"
    )
    digits.add_argument("--hidden-size", type=int, default=64)
    digits.add_argument("--test-per-class", type=int, default=40)
    digits.add_argument("--passes", type=int, default=1)
    digits.add_argument("--augmentation-copies", type=int, default=1)
    digits.add_argument("--seed", type=int, default=29)
    digits.add_argument("--output", type=str)
    memory = subparsers.add_parser(
        "memory", help="run Milestone 6 scalable-memory quality experiments"
    )
    memory.add_argument("--hidden-size", type=int, default=64)
    memory.add_argument("--test-per-class", type=int, default=40)
    memory.add_argument("--seed", type=int, default=29)
    memory.add_argument("--block-size", type=int, default=16)
    memory.add_argument(
        "--forgetting-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.9999, 0.999, 0.99, 0.95],
    )
    memory.add_argument("--output", type=str)
    scaling = subparsers.add_parser(
        "scale", help="run lazy image-stream and feature-width scaling benchmarks"
    )
    scaling.add_argument(
        "--sample-counts", type=int, nargs="+", default=[1_000, 10_000, 60_000]
    )
    scaling.add_argument("--image-sizes", type=int, nargs="+", default=[8, 28])
    scaling.add_argument("--hidden-size", type=int, default=64)
    scaling.add_argument(
        "--feature-widths", type=int, nargs="+", default=[65, 129, 257, 513]
    )
    scaling.add_argument("--feature-updates", type=int, default=1_000)
    scaling.add_argument(
        "--kinds",
        nargs="+",
        choices=(
            "lms",
            "rls",
            "diagonal_rls",
            "block_rls",
            "prototype",
            "protected",
        ),
        default=[
            "lms",
            "rls",
            "diagonal_rls",
            "block_rls",
            "prototype",
            "protected",
        ],
    )
    scaling.add_argument("--seed", type=int, default=41)
    scaling.add_argument("--output", type=str)
    replicate = subparsers.add_parser(
        "replicate", help="run delayed and continual benchmarks across seeds"
    )
    replicate.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11, 17, 23])
    replicate.add_argument("--output", type=str)
    plot = subparsers.add_parser("plot", help="plot a generated experiment JSON file")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "signal":
        config = SignalExperimentConfig(
            steps=args.steps,
            regime_length=args.regime_length,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        result = run_signal_experiment(config)
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "delayed":
        config = DelayedExperimentConfig(
            episodes=args.episodes,
            delay=args.delay,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        result = run_delayed_experiment(config)
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "continual":
        config = ContinualExperimentConfig(
            steps=args.steps,
            context_length=args.context_length,
            hidden_size=args.hidden_size,
            seed=args.seed,
        )
        result = run_continual_experiment(config)
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "digits":
        config = DigitsExperimentConfig(
            hidden_size=args.hidden_size,
            test_per_class=args.test_per_class,
            passes=args.passes,
            augmentation_copies=args.augmentation_copies,
            seed=args.seed,
        )
        result = run_digits_experiment(config)
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "memory":
        result = run_milestone6(
            Milestone6Config(
                hidden_size=args.hidden_size,
                test_per_class=args.test_per_class,
                seed=args.seed,
                block_size=args.block_size,
                forgetting_factors=tuple(args.forgetting_factors),
            )
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "scale":
        result = run_scaling_experiment(
            BlankImageScalingConfig(
                sample_counts=tuple(args.sample_counts),
                image_sizes=tuple(args.image_sizes),
                hidden_size=args.hidden_size,
                seed=args.seed,
                kinds=tuple(args.kinds),
            ),
            FeatureWidthScalingConfig(
                feature_widths=tuple(args.feature_widths),
                updates=args.feature_updates,
                seed=args.seed + 2,
                kinds=tuple(args.kinds),
            ),
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "replicate":
        result = run_replication(ReplicationConfig(seeds=tuple(args.seeds)))
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "plot":
        destination = plot_result(load_result(args.input), args.output)
        print(destination)
    return 0

"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
import json

from no_backprop.drift_suite import DriftSuiteConfig, run_drift_suite
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
from no_backprop.factor_free import FactorFreeMemoryConfig, run_factor_free_memory
from no_backprop.frontend_comparison import (
    FrontendComparisonConfig,
    PolarityComparisonConfig,
    PredictiveRepresentationConfig,
    PredictiveSurpriseComparisonConfig,
    run_frontend_comparison,
    run_polarity_comparison,
    run_predictive_representation_comparison,
    run_predictive_surprise_comparison,
)
from no_backprop.scaling import (
    BlankImageScalingConfig,
    FeatureWidthScalingConfig,
    MemoryCapacityScalingConfig,
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
    cumulative_memory = subparsers.add_parser(
        "cumulative-memory",
        help="run factor-free cumulative fast/slow memory experiments",
    )
    cumulative_memory.add_argument("--hidden-size", type=int, default=64)
    cumulative_memory.add_argument("--test-per-class", type=int, default=40)
    cumulative_memory.add_argument("--regularization", type=float, default=1.0)
    cumulative_memory.add_argument("--rank-bins", type=int, default=16)
    cumulative_memory.add_argument("--maturity-max-neurons", type=int, default=32)
    cumulative_memory.add_argument("--maturity-rbf-width", type=float, default=0.05)
    cumulative_memory.add_argument(
        "--maturity-min-center-distance", type=float, default=0.01
    )
    cumulative_memory.add_argument(
        "--key-prior-strength", type=float, default=4.0
    )
    cumulative_memory.add_argument(
        "--key-minimum-variance", type=float, default=4e-4
    )
    cumulative_memory.add_argument(
        "--key-maximum-variance", type=float, default=3.6e-3
    )
    cumulative_memory.add_argument("--seed", type=int, default=29)
    cumulative_memory.add_argument("--output", type=str)
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
    scaling.add_argument("--memory-neurons", type=int, default=32)
    scaling.add_argument("--memory-candidates", type=int, default=16)
    scaling.add_argument(
        "--key-capacities", type=int, nargs="+", default=[8, 16, 32, 64, 128]
    )
    scaling.add_argument(
        "--candidate-capacities", type=int, nargs="+", default=[4, 8, 16, 32, 64]
    )
    scaling.add_argument("--capacity-updates", type=int, default=1_000)
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
            "managed_probation",
        ),
        default=[
            "lms",
            "rls",
            "diagonal_rls",
            "block_rls",
            "prototype",
            "protected",
            "managed_probation",
        ],
    )
    scaling.add_argument("--seed", type=int, default=41)
    scaling.add_argument("--output", type=str)
    frontends = subparsers.add_parser(
        "frontends",
        help="compare recurrent, pixel, and fixed-convolution image frontends",
    )
    frontends.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    frontends.add_argument("--test-per-class", type=int, default=40)
    frontends.add_argument("--augmentation-copies", type=int, default=1)
    frontends.add_argument("--no-drift", action="store_true")
    frontends.add_argument("--output", type=str)
    predictive = subparsers.add_parser(
        "predictive",
        help="compare forward-only masked latent prediction with fixed controls",
    )
    predictive.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    predictive.add_argument("--test-per-class", type=int, default=40)
    predictive.add_argument("--augmentation-copies", type=int, default=1)
    predictive.add_argument("--predictor-regularization", type=float, default=1.0)
    predictive.add_argument("--no-drift", action="store_true")
    predictive.add_argument("--output", type=str)
    predictive_surprise = subparsers.add_parser(
        "predictive-surprise",
        help="test masked-prediction surprise as a recruitment signal",
    )
    predictive_surprise.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    predictive_surprise.add_argument("--test-per-class", type=int, default=40)
    predictive_surprise.add_argument(
        "--augmentation-copies", type=int, default=1
    )
    predictive_surprise.add_argument(
        "--predictor-regularization", type=float, default=1.0
    )
    predictive_surprise.add_argument("--no-drift", action="store_true")
    predictive_surprise.add_argument("--output", type=str)
    polarity = subparsers.add_parser(
        "polarity",
        help="compare polarity-aware convolution with recurrence",
    )
    polarity.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    polarity.add_argument("--test-per-class", type=int, default=40)
    polarity.add_argument("--augmentation-copies", type=int, default=1)
    polarity.add_argument("--no-drift", action="store_true")
    polarity.add_argument("--output", type=str)
    drift_suite = subparsers.add_parser(
        "drift-suite",
        help="run recurring label-preserving image transformations",
    )
    drift_suite.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    drift_suite.add_argument("--test-per-class", type=int, default=40)
    drift_suite.add_argument(
        "--transformations",
        nargs="+",
        choices=(
            "inversion",
            "low_contrast",
            "gaussian_noise",
            "center_occlusion",
            "translation",
            "striped_background",
        ),
        default=[
            "inversion",
            "low_contrast",
            "gaussian_noise",
            "center_occlusion",
            "translation",
            "striped_background",
        ],
    )
    drift_suite.add_argument("--output", type=str)
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
    elif args.command == "cumulative-memory":
        result = run_factor_free_memory(
            FactorFreeMemoryConfig(
                hidden_size=args.hidden_size,
                test_per_class=args.test_per_class,
                regularization=args.regularization,
                rank_bins=args.rank_bins,
                maturity_max_neurons=args.maturity_max_neurons,
                maturity_rbf_width=args.maturity_rbf_width,
                maturity_min_center_distance=args.maturity_min_center_distance,
                key_prior_strength=args.key_prior_strength,
                key_minimum_variance=args.key_minimum_variance,
                key_maximum_variance=args.key_maximum_variance,
                seed=args.seed,
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
                memory_neurons=args.memory_neurons,
                memory_candidates=args.memory_candidates,
                seed=args.seed,
                kinds=tuple(args.kinds),
            ),
            FeatureWidthScalingConfig(
                feature_widths=tuple(args.feature_widths),
                updates=args.feature_updates,
                memory_neurons=args.memory_neurons,
                memory_candidates=args.memory_candidates,
                seed=args.seed + 2,
                kinds=tuple(args.kinds),
            ),
            MemoryCapacityScalingConfig(
                key_capacities=tuple(args.key_capacities),
                candidate_capacities=tuple(args.candidate_capacities),
                fixed_key_capacity=args.memory_neurons,
                fixed_candidate_capacity=args.memory_candidates,
                updates=args.capacity_updates,
                seed=args.seed + 6,
            ),
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "frontends":
        result = run_frontend_comparison(
            FrontendComparisonConfig(
                seeds=tuple(args.seeds),
                test_per_class=args.test_per_class,
                augmentation_copies=args.augmentation_copies,
                include_drift=not args.no_drift,
            )
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "predictive":
        result = run_predictive_representation_comparison(
            PredictiveRepresentationConfig(
                seeds=tuple(args.seeds),
                test_per_class=args.test_per_class,
                augmentation_copies=args.augmentation_copies,
                predictor_regularization=args.predictor_regularization,
                include_drift=not args.no_drift,
            )
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "predictive-surprise":
        result = run_predictive_surprise_comparison(
            PredictiveSurpriseComparisonConfig(
                seeds=tuple(args.seeds),
                test_per_class=args.test_per_class,
                augmentation_copies=args.augmentation_copies,
                predictor_regularization=args.predictor_regularization,
                include_drift=not args.no_drift,
            )
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "polarity":
        result = run_polarity_comparison(
            PolarityComparisonConfig(
                seeds=tuple(args.seeds),
                test_per_class=args.test_per_class,
                augmentation_copies=args.augmentation_copies,
                include_drift=not args.no_drift,
            )
        )
        if args.output:
            write_json_result(result, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "drift-suite":
        result = run_drift_suite(
            DriftSuiteConfig(
                seeds=tuple(args.seeds),
                test_per_class=args.test_per_class,
                transformations=tuple(args.transformations),
            )
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

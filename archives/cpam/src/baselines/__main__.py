"""CLI for comparisons that are intentionally outside the core package."""

from __future__ import annotations

import argparse
import json

from baselines.compare import (
    DigitsSystemsComparisonConfig,
    SystemsComparisonConfig,
    run_digits_systems_comparison,
    run_systems_comparison,
)
from baselines.analytic_memory import (
    AnalyticMemoryComparisonConfig,
    run_analytic_memory_comparison,
)
from baselines.memory_backprop import (
    MemoryBackpropComparisonConfig,
    run_memory_backprop_comparison,
)
from no_backprop.experiment import write_json_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run conventional baseline comparisons")
    parser.add_argument(
        "--benchmark",
        choices=("signal", "digits", "memory-capstone", "analytic-memory"),
        default="signal",
    )
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--regime-length", type=int, default=750)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--augmentation-copies", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 17, 23, 29, 37, 41, 47, 53],
    )
    parser.add_argument(
        "--development-seeds", type=int, nargs="+", default=[2, 5, 13]
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.0001, 0.0003, 0.001, 0.003, 0.01],
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args(argv)
    if args.benchmark == "analytic-memory":
        result = run_analytic_memory_comparison(
            AnalyticMemoryComparisonConfig(
                test_seeds=tuple(args.test_seeds),
                development_seeds=tuple(args.development_seeds),
                test_per_class=args.test_per_class,
            )
        )
    elif args.benchmark == "memory-capstone":
        result = run_memory_backprop_comparison(
            MemoryBackpropComparisonConfig(
                test_seeds=tuple(args.test_seeds),
                development_seeds=tuple(args.development_seeds),
                test_per_class=args.test_per_class,
                learning_rates=tuple(args.learning_rates),
            )
        )
    elif args.benchmark == "digits":
        seed = 29 if args.seed is None else args.seed
        digits_config = DigitsSystemsComparisonConfig(
            hidden_size=args.hidden_size,
            test_per_class=args.test_per_class,
            passes=args.passes,
            augmentation_copies=args.augmentation_copies,
            seed=seed,
            bptt_learning_rate=args.learning_rate,
        )
        result = run_digits_systems_comparison(digits_config)
    else:
        seed = 7 if args.seed is None else args.seed
        signal_config = SystemsComparisonConfig(
            steps=args.steps,
            regime_length=args.regime_length,
            hidden_size=args.hidden_size,
            seed=seed,
            bptt_windows=tuple(args.windows),
        )
        result = run_systems_comparison(signal_config)
    if args.output:
        write_json_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line entry point for reproducible experiments."""

from __future__ import annotations

import argparse
import json

from no_backprop.experiment import (
    SignalExperimentConfig,
    run_signal_experiment,
    write_json_result,
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
    return 0

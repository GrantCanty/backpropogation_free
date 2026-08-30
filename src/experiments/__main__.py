"""CLI for active method-neutral synthetic-stream experiments."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Mapping

from continual_core.plotting import plot_result
from continual_core.results import write_json_result
from experiments.solver_comparison import (
    SolverComparisonConfig,
    run_solver_comparison,
    write_solver_artifacts,
)
from experiments.streams import (
    ContinualExperimentConfig,
    DelayedExperimentConfig,
    SignalExperimentConfig,
    run_continual_experiment,
    run_delayed_experiment,
    run_signal_experiment,
)


ExperimentConfig = (
    SignalExperimentConfig
    | DelayedExperimentConfig
    | ContinualExperimentConfig
    | SolverComparisonConfig
)
CONFIG_TYPES = {
    "signal": SignalExperimentConfig,
    "delayed": DelayedExperimentConfig,
    "continual": ContinualExperimentConfig,
    "solver_comparison": SolverComparisonConfig,
}


def read_config(path: str | Path) -> tuple[str, dict[str, Any]]:
    """Read a self-identifying JSON experiment configuration."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("configuration must be a JSON object")
    benchmark = document.pop("benchmark", None)
    if benchmark not in CONFIG_TYPES:
        choices = ", ".join(CONFIG_TYPES)
        raise ValueError(f"configuration benchmark must be one of: {choices}")
    return benchmark, document


def build_config(
    benchmark: str,
    values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Validate JSON keys and apply explicit command-line overrides."""

    config_type = CONFIG_TYPES[benchmark]
    permitted = {field.name for field in fields(config_type)}
    parameters = dict(values or {})
    unknown = sorted(parameters.keys() - permitted)
    if unknown:
        raise ValueError(
            f"unknown {benchmark} configuration keys: {', '.join(unknown)}"
        )
    for name, value in (overrides or {}).items():
        if value is not None and name in permitted:
            parameters[name] = value
    return config_type(**parameters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run active method-neutral continual-learning experiments"
    )
    parser.add_argument("--benchmark", choices=tuple(CONFIG_TYPES))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--regime-length", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--delay", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plot",
        type=Path,
        help="write a plot for supported benchmarks (currently signal)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv == []:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    values: dict[str, Any] = {}
    benchmark = args.benchmark
    if args.config is not None:
        try:
            config_benchmark, values = read_config(args.config)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        if benchmark is not None and benchmark != config_benchmark:
            parser.error(
                f"--benchmark {benchmark!r} conflicts with config benchmark "
                f"{config_benchmark!r}"
            )
        benchmark = config_benchmark
    benchmark = benchmark or "signal"
    overrides = {
        "steps": args.steps,
        "regime_length": args.regime_length,
        "episodes": args.episodes,
        "delay": args.delay,
        "context_length": args.context_length,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
    }
    try:
        config = build_config(benchmark, values, overrides)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.plot is not None and benchmark != "signal":
        parser.error("--plot is currently supported only for the signal benchmark")

    if benchmark == "solver_comparison":
        result = run_solver_comparison(config)  # type: ignore[arg-type]
    elif benchmark == "delayed":
        result = run_delayed_experiment(config)
    elif benchmark == "continual":
        result = run_continual_experiment(config)
    else:
        result = run_signal_experiment(config)
    if args.output is not None and benchmark == "solver_comparison":
        write_solver_artifacts(result, args.output)
    elif args.output is not None:
        write_json_result(result, args.output)
    if args.plot is not None:
        try:
            plot_result(result, args.plot)
        except ValueError as error:
            parser.error(str(error))
    displayed = (
        {
            "experiment": result["experiment"],
            "selected_hyperparameters": result["selected_hyperparameters"],
            "findings": result["findings"],
            "artifacts": str(args.output) if args.output is not None else None,
        }
        if benchmark == "solver_comparison" and args.output is not None
        else result
    )
    print(json.dumps(displayed, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

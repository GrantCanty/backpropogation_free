"""CLI for comparisons that are intentionally outside the core package."""

from __future__ import annotations

import argparse
import json

from baselines.compare import SystemsComparisonConfig, run_systems_comparison
from no_backprop.experiment import write_json_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run conventional baseline comparisons")
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--regime-length", type=int, default=750)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--windows", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--output", type=str)
    args = parser.parse_args(argv)
    config = SystemsComparisonConfig(
        steps=args.steps,
        regime_length=args.regime_length,
        hidden_size=args.hidden_size,
        seed=args.seed,
        bptt_windows=tuple(args.windows),
    )
    result = run_systems_comparison(config)
    if args.output:
        write_json_result(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

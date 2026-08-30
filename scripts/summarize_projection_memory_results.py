#!/usr/bin/env python3
"""Generate JSON, Markdown, and LaTeX tables from raw final-study artifacts."""

from pathlib import Path
import argparse
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)

from experiments.projection_memory_reporting import (
    summarize_projection_results,
    write_projection_summary,
)


def _formats(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="dense_exact")
    parser.add_argument("--formats", type=_formats, default=("json", "md", "tex"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_projection_results(args.input, baseline=args.baseline)
    written = write_projection_summary(summary, args.output, args.formats)
    for name, path in written.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

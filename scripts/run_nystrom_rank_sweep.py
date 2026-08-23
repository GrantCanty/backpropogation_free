#!/usr/bin/env python3
"""Public entry point for the dense/sparse Nyström rank sweep."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.run_sparse_nystrom_rank_sweep import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())

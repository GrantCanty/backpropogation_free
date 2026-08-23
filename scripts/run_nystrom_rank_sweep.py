#!/usr/bin/env python3
"""Public entry point for the dense/sparse Nyström rank sweep."""

from scripts.run_sparse_nystrom_rank_sweep import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run fresh-seed confirmation for the selected Frequent-Directions candidate."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.run_frequent_directions_sweep import main


if __name__ == "__main__":
    raise SystemExit(main(["--phase", "confirmation", *sys.argv[1:]]))

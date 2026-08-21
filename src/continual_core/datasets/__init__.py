"""Dataset loading and deterministic protocol construction."""

from continual_core.datasets.digits import (
    DigitsProtocol,
    DigitsSegment,
    DigitsSplit,
    augment_digits_split,
    build_digits_segments,
    load_digits_split,
)

__all__ = [
    "DigitsProtocol",
    "DigitsSegment",
    "DigitsSplit",
    "augment_digits_split",
    "build_digits_segments",
    "load_digits_split",
]

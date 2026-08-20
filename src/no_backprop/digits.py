"""Local 8x8 handwritten-digit data and deterministic stream orderings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DigitsProtocol = Literal["shuffled", "class_ordered"]


@dataclass(frozen=True)
class DigitsSplit:
    """A stratified train/test split of scikit-learn's bundled digits data."""

    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class DigitsSegment:
    """One checkpoint-sized portion of an online training stream."""

    pass_index: int
    segment_index: int
    focus_class: int | None
    indices: np.ndarray


def load_digits_split(*, test_per_class: int = 40, seed: int = 0) -> DigitsSplit:
    """Load a deterministic split without downloading any data.

    ``sklearn.datasets.load_digits`` is packaged with scikit-learn. Importing
    and loading it performs no network request.
    """

    if test_per_class <= 0:
        raise ValueError("test_per_class must be positive")
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "the digits benchmark needs an existing scikit-learn installation; "
            "it will not attempt to install or download anything"
        ) from exc

    images, labels = load_digits(return_X_y=True)
    images = np.asarray(images, dtype=np.float64).reshape(-1, 8, 8) / 16.0
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    for class_index in np.unique(labels):
        candidates = np.flatnonzero(labels == class_index)
        if test_per_class >= len(candidates):
            raise ValueError(
                f"test_per_class must be smaller than class {class_index} size "
                f"({len(candidates)})"
            )
        shuffled = rng.permutation(candidates)
        test_indices.append(shuffled[:test_per_class])
        train_indices.append(shuffled[test_per_class:])

    train = np.concatenate(train_indices)
    test = np.concatenate(test_indices)
    return DigitsSplit(
        train_images=images[train],
        train_labels=labels[train],
        test_images=images[test],
        test_labels=labels[test],
    )


def build_digits_segments(
    labels: np.ndarray,
    *,
    protocol: DigitsProtocol,
    passes: int = 1,
    seed: int = 0,
) -> list[DigitsSegment]:
    """Build shuffled or class-ordered segments over a fixed training split.

    Both protocols contain every training example exactly once per pass. The
    shuffled stream is divided into ten checkpoint segments; the class-ordered
    stream has one segment per class and is shuffled only within each class.
    """

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty vector")
    if protocol not in ("shuffled", "class_ordered"):
        raise ValueError(f"unknown digits protocol: {protocol}")
    if passes <= 0:
        raise ValueError("passes must be positive")

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    segments: list[DigitsSegment] = []
    segment_index = 0
    for pass_index in range(passes):
        if protocol == "shuffled":
            chunks = np.array_split(rng.permutation(len(labels)), len(classes))
            pass_segments = [(None, chunk) for chunk in chunks]
        else:
            pass_segments = [
                (int(class_index), rng.permutation(np.flatnonzero(labels == class_index)))
                for class_index in classes
            ]
        for focus_class, indices in pass_segments:
            segments.append(
                DigitsSegment(
                    pass_index=pass_index,
                    segment_index=segment_index,
                    focus_class=focus_class,
                    indices=np.asarray(indices, dtype=np.int64),
                )
            )
            segment_index += 1
    return segments

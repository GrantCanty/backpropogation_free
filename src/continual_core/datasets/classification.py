"""Explicit local/downloaded image-classification dataset sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from continual_core.datasets.digits import load_digits_split


@dataclass(frozen=True)
class ClassificationSplit:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray


def _normalize_labels(
    train_labels: np.ndarray, test_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    classes = np.unique(np.concatenate((train, test)))
    mapping = {str(label): index for index, label in enumerate(classes)}
    normalized_train = np.asarray(
        [mapping[str(label)] for label in train], dtype=np.int64
    )
    normalized_test = np.asarray(
        [mapping[str(label)] for label in test], dtype=np.int64
    )
    return normalized_train, normalized_test


def _limit_test_examples(
    split: ClassificationSplit, *, per_class: int, seed: int
) -> ClassificationSplit:
    if per_class <= 0:
        raise ValueError("test_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in np.unique(split.test_labels):
        indices = np.flatnonzero(split.test_labels == label)
        if per_class > len(indices):
            raise ValueError(
                f"test_per_class exceeds available examples for class {label}"
            )
        selected.append(rng.permutation(indices)[:per_class])
    test_indices = np.concatenate(selected)
    return ClassificationSplit(
        train_images=split.train_images,
        train_labels=split.train_labels,
        test_images=split.test_images[test_indices],
        test_labels=split.test_labels[test_indices],
    )


def _validate_split(split: ClassificationSplit) -> ClassificationSplit:
    train_images = np.asarray(split.train_images, dtype=np.float64)
    test_images = np.asarray(split.test_images, dtype=np.float64)
    train_labels, test_labels = _normalize_labels(
        split.train_labels, split.test_labels
    )
    if train_images.ndim < 2 or test_images.ndim != train_images.ndim:
        raise ValueError("train and test images must have matching sample shapes")
    if train_images.shape[1:] != test_images.shape[1:]:
        raise ValueError("train and test observation shapes do not match")
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError("image and label counts do not match")
    if len(np.unique(train_labels)) < 2:
        raise ValueError("classification datasets need at least two classes")
    maximum = max(float(np.max(train_images)), float(np.max(test_images)))
    minimum = min(float(np.min(train_images)), float(np.min(test_images)))
    if maximum > 1.0 or minimum < 0.0:
        scale = 255.0 if maximum <= 255.0 and minimum >= 0.0 else maximum - minimum
        if scale <= 0.0:
            raise ValueError("dataset observations have no usable numeric range")
        train_images = (train_images - minimum) / scale
        test_images = (test_images - minimum) / scale
    return ClassificationSplit(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def load_classification_split(
    dataset: str,
    *,
    test_per_class: int,
    seed: int,
    dataset_path: str | Path | None = None,
    allow_download: bool = False,
    cache_directory: str | Path | None = None,
) -> ClassificationSplit:
    """Load a declared dataset without implicit network access.

    ``digits`` is bundled with scikit-learn. ``npz`` expects the keys
    ``train_images``, ``train_labels``, ``test_images``, and ``test_labels``.
    ``fashion_mnist`` uses OpenML and refuses to run unless downloading was
    explicitly enabled by the caller.
    """

    if dataset == "digits":
        source = load_digits_split(test_per_class=test_per_class, seed=seed)
        return _validate_split(ClassificationSplit(**source.__dict__))
    if dataset == "npz":
        if dataset_path is None:
            raise ValueError("dataset_path is required when dataset='npz'")
        with np.load(Path(dataset_path), allow_pickle=False) as archive:
            required = {
                "train_images",
                "train_labels",
                "test_images",
                "test_labels",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    "npz dataset is missing keys: " + ", ".join(sorted(missing))
                )
            split = ClassificationSplit(
                **{name: np.asarray(archive[name]) for name in required}
            )
        return _limit_test_examples(
            _validate_split(split), per_class=test_per_class, seed=seed
        )
    if dataset == "fashion_mnist":
        if not allow_download:
            raise RuntimeError(
                "Fashion-MNIST requires network access; rerun with the explicit "
                "--allow-download option on the cloud provider"
            )
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Fashion-MNIST requires an existing scikit-learn installation"
            ) from exc
        kwargs: dict[str, object] = {
            "name": "Fashion-MNIST",
            "version": 1,
            "as_frame": False,
        }
        if cache_directory is not None:
            kwargs["data_home"] = str(cache_directory)
        try:
            features, labels = fetch_openml(parser="auto", return_X_y=True, **kwargs)
        except TypeError:  # scikit-learn before the parser argument
            features, labels = fetch_openml(return_X_y=True, **kwargs)
        images = np.asarray(features, dtype=np.float64).reshape(-1, 28, 28)
        labels = np.asarray(labels)
        split = ClassificationSplit(
            train_images=images[:60_000],
            train_labels=labels[:60_000],
            test_images=images[60_000:],
            test_labels=labels[60_000:],
        )
        return _limit_test_examples(
            _validate_split(split), per_class=test_per_class, seed=seed
        )
    raise ValueError("dataset must be one of: digits, fashion_mnist, npz")


def augment_image_split(
    split: ClassificationSplit,
    *,
    copies: int,
    max_shift: int,
    noise_std: float,
    seed: int,
) -> ClassificationSplit:
    """Apply the existing deterministic image augmentation to any 2D images."""

    if copies < 0 or max_shift < 0 or noise_std < 0.0:
        raise ValueError("augmentation values cannot be negative")
    if copies == 0:
        return split
    if split.train_images.ndim != 3:
        raise ValueError("augmentation requires observations shaped [N, H, W]")
    rng = np.random.default_rng(seed)
    image_groups = [split.train_images]
    label_groups = [split.train_labels]
    for _ in range(copies):
        augmented = np.empty_like(split.train_images)
        for index, image in enumerate(split.train_images):
            row_shift = int(rng.integers(-max_shift, max_shift + 1))
            column_shift = int(rng.integers(-max_shift, max_shift + 1))
            shifted = np.roll(image, (row_shift, column_shift), axis=(0, 1))
            if row_shift > 0:
                shifted[:row_shift, :] = 0.0
            elif row_shift < 0:
                shifted[row_shift:, :] = 0.0
            if column_shift > 0:
                shifted[:, :column_shift] = 0.0
            elif column_shift < 0:
                shifted[:, column_shift:] = 0.0
            augmented[index] = np.clip(
                shifted + rng.normal(0.0, noise_std, shifted.shape), 0.0, 1.0
            )
        image_groups.append(augmented)
        label_groups.append(split.train_labels.copy())
    return ClassificationSplit(
        train_images=np.concatenate(image_groups),
        train_labels=np.concatenate(label_groups),
        test_images=split.test_images,
        test_labels=split.test_labels,
    )

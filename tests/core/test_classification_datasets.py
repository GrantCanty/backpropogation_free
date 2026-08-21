import numpy as np
import pytest

from continual_core.datasets.classification import load_classification_split


def test_fashion_mnist_refuses_implicit_download() -> None:
    with pytest.raises(RuntimeError, match="--allow-download"):
        load_classification_split(
            "fashion_mnist", test_per_class=2, seed=3, allow_download=False
        )


def test_npz_dataset_is_normalized_remapped_and_limited(tmp_path) -> None:
    train_images = np.arange(16 * 4, dtype=np.uint8).reshape(16, 2, 2)
    train_labels = np.asarray([10, 20] * 8)
    test_images = np.arange(8 * 4, dtype=np.uint8).reshape(8, 2, 2)
    test_labels = np.asarray([10, 20] * 4)
    path = tmp_path / "images.npz"
    np.savez(
        path,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )
    split = load_classification_split(
        "npz", dataset_path=path, test_per_class=2, seed=5
    )
    assert split.train_images.shape == (16, 2, 2)
    assert split.test_images.shape == (4, 2, 2)
    assert set(split.train_labels) == {0, 1}
    assert set(split.test_labels) == {0, 1}
    assert 0.0 <= float(np.min(split.train_images))
    assert float(np.max(split.train_images)) <= 1.0

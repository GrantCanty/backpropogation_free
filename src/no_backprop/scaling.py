"""Synthetic systems benchmarks for stream length and feature-width scaling."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from no_backprop.readouts import (
    BlockRLSReadout,
    DiagonalRLSReadout,
    LMSReadout,
    ProtectedFastSlowReadout,
    PrototypeReadout,
    RLSReadout,
)
from no_backprop.reservoir import OnlineReservoir, ReservoirConfig


ScalingKind = Literal[
    "lms",
    "rls",
    "diagonal_rls",
    "block_rls",
    "prototype",
    "protected",
]


DEFAULT_SCALING_KINDS: tuple[ScalingKind, ...] = (
    "lms",
    "rls",
    "diagonal_rls",
    "block_rls",
    "prototype",
    "protected",
)


@dataclass(frozen=True)
class BlankImageScalingConfig:
    """End-to-end scaling without allocating or downloading an image dataset."""

    sample_counts: tuple[int, ...] = (1_000, 10_000, 60_000)
    image_sizes: tuple[int, ...] = (8, 28)
    hidden_size: int = 64
    classes: int = 10
    block_size: int = 16
    seed: int = 41
    kinds: tuple[ScalingKind, ...] = DEFAULT_SCALING_KINDS


@dataclass(frozen=True)
class FeatureWidthScalingConfig:
    """Readout-only measurements that isolate RLS feature-width complexity."""

    feature_widths: tuple[int, ...] = (65, 129, 257, 513)
    updates: int = 1_000
    classes: int = 10
    block_size: int = 16
    seed: int = 43
    kinds: tuple[ScalingKind, ...] = DEFAULT_SCALING_KINDS
    projected_widths: tuple[int, ...] = (65, 129, 257, 513, 1_025, 2_049, 4_097)


def _build_readout(
    kind: ScalingKind,
    feature_size: int,
    classes: int,
    *,
    block_size: int,
    seed: int,
):
    if kind == "lms":
        return LMSReadout(
            feature_size, classes, seed=seed, learning_rate=0.18
        )
    if kind == "rls":
        return RLSReadout(
            feature_size,
            classes,
            seed=seed,
            regularization=1.0,
            forgetting_factor=1.0,
        )
    if kind == "diagonal_rls":
        return DiagonalRLSReadout(
            feature_size,
            classes,
            seed=seed,
            regularization=1.0,
            forgetting_factor=1.0,
        )
    if kind == "block_rls":
        return BlockRLSReadout(
            feature_size,
            classes,
            seed=seed,
            regularization=1.0,
            forgetting_factor=1.0,
            block_size=block_size,
        )
    if kind == "prototype":
        return PrototypeReadout(feature_size, classes, seed=seed)
    if kind == "protected":
        return ProtectedFastSlowReadout(feature_size, classes, seed=seed)
    raise ValueError(f"unknown scaling learner: {kind}")


def _build_image_learner(
    kind: ScalingKind, config: BlankImageScalingConfig, image_size: int
) -> OnlineReservoir:
    reservoir = ReservoirConfig(
        input_size=image_size,
        hidden_size=config.hidden_size,
        output_size=config.classes,
        spectral_radius=0.88,
        input_scale=0.65,
        leak_rate=0.7,
        seed=config.seed,
    )
    readout = _build_readout(
        kind,
        config.hidden_size + 1,
        config.classes,
        block_size=config.block_size,
        seed=config.seed,
    )
    return OnlineReservoir(reservoir, readout)


def run_blank_image_model(
    kind: ScalingKind,
    *,
    image_size: int,
    samples: int,
    config: BlankImageScalingConfig,
) -> dict[str, Any]:
    """Process a lazily reused blank image and discard all predictions."""

    if image_size <= 0 or samples <= 0:
        raise ValueError("image_size and samples must be positive")
    learner = _build_image_learner(kind, config, image_size)
    blank_image = np.zeros((image_size, image_size), dtype=np.float64)
    no_feedback = np.full(config.classes, np.nan, dtype=np.float64)
    targets = np.eye(config.classes, dtype=np.float64)
    state_before = learner.state_nbytes
    started = time.perf_counter()
    for sample_index in range(samples):
        learner.reset_state()
        for row_index, row in enumerate(blank_image):
            learner.predict(row)
            target = (
                targets[sample_index % config.classes]
                if row_index == image_size - 1
                else no_feedback
            )
            learner.learn(target)
    elapsed = time.perf_counter() - started
    return {
        "model": kind,
        "image_size": [image_size, image_size],
        "samples": samples,
        "row_events": samples * image_size,
        "elapsed_seconds": elapsed,
        "images_per_second": samples / elapsed,
        "row_events_per_second": samples * image_size / elapsed,
        "state_bytes_before": state_before,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": state_before == learner.state_nbytes,
        "dataset_storage_bytes": blank_image.nbytes,
    }


def run_blank_image_scaling(
    config: BlankImageScalingConfig = BlankImageScalingConfig(),
) -> dict[str, Any]:
    if not config.sample_counts or not config.image_sizes or not config.kinds:
        raise ValueError("sample counts, image sizes, and learners cannot be empty")
    return {
        "experiment": "lazy_blank_image_scaling",
        "config": asdict(config),
        "runs": [
            run_blank_image_model(
                kind, image_size=image_size, samples=samples, config=config
            )
            for image_size in config.image_sizes
            for samples in config.sample_counts
            for kind in config.kinds
        ],
    }


def _projected_readout_bytes(
    kind: ScalingKind, width: int, classes: int, block_size: int
) -> int:
    itemsize = np.dtype(np.float64).itemsize
    weights = classes * width
    if kind == "rls":
        auxiliary = width * width
    elif kind == "diagonal_rls":
        auxiliary = width
    elif kind == "block_rls":
        full_blocks, remainder = divmod(width, block_size)
        auxiliary = full_blocks * block_size * block_size + remainder * remainder
    elif kind == "prototype":
        weights = classes * width
        auxiliary = classes
    elif kind == "protected":
        weights = 2 * classes * width
        auxiliary = classes
    else:
        auxiliary = 0
    return itemsize * (weights + auxiliary)


def run_feature_width_scaling(
    config: FeatureWidthScalingConfig = FeatureWidthScalingConfig(),
) -> dict[str, Any]:
    """Measure update throughput and report analytic state at wider dimensions."""

    if config.updates <= 0 or not config.feature_widths or not config.kinds:
        raise ValueError("updates, feature widths, and learners must be positive")
    rng = np.random.default_rng(config.seed)
    measured: list[dict[str, Any]] = []
    for width in config.feature_widths:
        features = rng.normal(size=(config.updates, width))
        features /= np.maximum(
            np.linalg.norm(features, axis=1, keepdims=True),
            np.finfo(float).tiny,
        )
        targets = np.eye(config.classes, dtype=np.float64)[
            np.arange(config.updates) % config.classes
        ]
        for kind in config.kinds:
            readout = _build_readout(
                kind,
                width,
                config.classes,
                block_size=config.block_size,
                seed=config.seed,
            )
            state_before = readout.state_nbytes
            started = time.perf_counter()
            for feature, target in zip(features, targets):
                prediction = readout.predict(feature)
                readout.update(feature, target, prediction)
            elapsed = time.perf_counter() - started
            measured.append(
                {
                    "model": kind,
                    "feature_width": width,
                    "updates": config.updates,
                    "elapsed_seconds": elapsed,
                    "updates_per_second": config.updates / elapsed,
                    "state_bytes_before": state_before,
                    "state_bytes_after": readout.state_nbytes,
                    "bounded_state": state_before == readout.state_nbytes,
                }
            )
    projected = [
        {
            "model": kind,
            "feature_width": width,
            "projected_state_bytes": _projected_readout_bytes(
                kind, width, config.classes, config.block_size
            ),
        }
        for width in config.projected_widths
        for kind in config.kinds
    ]
    return {
        "experiment": "readout_feature_width_scaling",
        "config": asdict(config),
        "measured": measured,
        "projected": projected,
    }


def run_scaling_experiment(
    image_config: BlankImageScalingConfig = BlankImageScalingConfig(),
    feature_config: FeatureWidthScalingConfig = FeatureWidthScalingConfig(),
) -> dict[str, Any]:
    return {
        "experiment": "milestone_6_scaling",
        "blank_images": run_blank_image_scaling(image_config),
        "feature_widths": run_feature_width_scaling(feature_config),
    }

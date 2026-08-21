"""No-download method-neutral resource-scaling experiments."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from baselines.readouts import (
    BlockRLSReadout,
    DiagonalRLSReadout,
    LMSReadout,
    ProtectedFastSlowReadout,
    PrototypeReadout,
    RLSReadout,
)
from baselines.reservoir import OnlineReservoir, ReservoirConfig
from methods.cpam.readouts import (
    CumulativeMaturityReadout,
    ManagedProbationaryMaturityReadout,
)


ScalingKind = Literal[
    "lms",
    "rls",
    "diagonal_rls",
    "block_rls",
    "prototype",
    "protected",
    "managed_probation",
]


DEFAULT_SCALING_KINDS: tuple[ScalingKind, ...] = (
    "lms",
    "rls",
    "diagonal_rls",
    "block_rls",
    "prototype",
    "protected",
    "managed_probation",
)


@dataclass(frozen=True)
class BlankImageScalingConfig:
    """End-to-end scaling without allocating or downloading an image dataset."""

    sample_counts: tuple[int, ...] = (1_000, 10_000, 60_000)
    image_sizes: tuple[int, ...] = (8, 28)
    hidden_size: int = 64
    classes: int = 10
    block_size: int = 16
    memory_neurons: int = 32
    memory_candidates: int = 16
    seed: int = 41
    kinds: tuple[ScalingKind, ...] = DEFAULT_SCALING_KINDS


@dataclass(frozen=True)
class FeatureWidthScalingConfig:
    """Readout-only measurements that isolate RLS feature-width complexity."""

    feature_widths: tuple[int, ...] = (65, 129, 257, 513)
    updates: int = 1_000
    classes: int = 10
    block_size: int = 16
    memory_neurons: int = 32
    memory_candidates: int = 16
    seed: int = 43
    kinds: tuple[ScalingKind, ...] = DEFAULT_SCALING_KINDS
    projected_widths: tuple[int, ...] = (65, 129, 257, 513, 1_025, 2_049, 4_097)


@dataclass(frozen=True)
class MemoryCapacityScalingConfig:
    """Managed-memory cost as active-key and candidate bounds increase."""

    feature_width: int = 65
    key_capacities: tuple[int, ...] = (8, 16, 32, 64, 128)
    candidate_capacities: tuple[int, ...] = (4, 8, 16, 32, 64)
    fixed_key_capacity: int = 32
    fixed_candidate_capacity: int = 16
    updates: int = 1_000
    classes: int = 10
    seed: int = 47
    projected_widths: tuple[int, ...] = (65, 129, 257, 513, 1_025, 2_049, 4_097)
    projected_key_capacities: tuple[int, ...] = (32, 64, 128, 256, 512)


def _build_readout(
    kind: ScalingKind,
    feature_size: int,
    classes: int,
    *,
    block_size: int,
    seed: int,
    memory_neurons: int = 32,
    memory_candidates: int = 16,
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
    if kind == "managed_probation":
        return ManagedProbationaryMaturityReadout(
            feature_size,
            classes,
            seed=seed,
            regularization=1.0,
            max_neurons=memory_neurons,
            max_candidates=memory_candidates,
        )
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
        memory_neurons=config.memory_neurons,
        memory_candidates=config.memory_candidates,
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
        "downloaded_data_bytes": 0,
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


def _projected_managed_memory_bytes(
    width: int, classes: int, key_capacity: int, candidate_capacity: int
) -> int:
    """Return exact allocated array bytes without constructing the readout."""

    if width <= 0 or classes <= 0 or min(key_capacity, candidate_capacity) < 0:
        raise ValueError(
            "managed-memory widths must be positive and capacities nonnegative"
        )
    expanded = width + key_capacity
    base = (
        classes * expanded
        + expanded * expanded
        + key_capacity * width
        + 4 * key_capacity
        + 11
    )
    probation = candidate_capacity * width + 4 * candidate_capacity + 5
    return np.dtype(np.float64).itemsize * (base + probation)


def _projected_readout_bytes(
    kind: ScalingKind,
    width: int,
    classes: int,
    block_size: int,
    memory_neurons: int,
    memory_candidates: int,
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
    elif kind == "managed_probation":
        return _projected_managed_memory_bytes(
            width, classes, memory_neurons, memory_candidates
        )
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
                memory_neurons=config.memory_neurons,
                memory_candidates=config.memory_candidates,
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
                kind,
                width,
                config.classes,
                config.block_size,
                config.memory_neurons,
                config.memory_candidates,
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


def _measure_managed_memory(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    key_capacity: int,
    candidate_capacity: int,
    classes: int,
    seed: int,
    prefill_keys: bool,
) -> dict[str, Any]:
    readout = ManagedProbationaryMaturityReadout(
        features.shape[1],
        classes,
        seed=seed,
        regularization=1.0,
        max_neurons=key_capacity,
        max_candidates=candidate_capacity,
    )
    if prefill_keys:
        # Capacity timing needs the local-activation work even though random
        # samples rarely confirm probationary keys on their own.
        for index in range(key_capacity):
            center = features[index % len(features)]
            CumulativeMaturityReadout._recruit(
                readout, center, target_class=index % classes
            )
    state_before = readout.state_nbytes
    started = time.perf_counter()
    for feature, target in zip(features, targets):
        prediction = readout.predict(feature)
        readout.update(feature, target, prediction)
    elapsed = time.perf_counter() - started
    return {
        "feature_width": features.shape[1],
        "key_capacity": key_capacity,
        "candidate_capacity": candidate_capacity,
        "updates": len(features),
        "elapsed_seconds": elapsed,
        "updates_per_second": len(features) / elapsed,
        "state_bytes_before": state_before,
        "state_bytes_after": readout.state_nbytes,
        "bounded_state": state_before == readout.state_nbytes,
        "active_keys": readout.diagnostics["active_neurons"],
        "pending_candidates": readout.diagnostics["pending_candidates"],
        "prefilled_keys": prefill_keys,
    }


def run_memory_capacity_scaling(
    config: MemoryCapacityScalingConfig = MemoryCapacityScalingConfig(),
) -> dict[str, Any]:
    """Measure each memory-capacity axis and project their combined state."""

    if (
        config.feature_width <= 0
        or config.updates <= 0
        or config.classes <= 0
        or not config.key_capacities
        or not config.candidate_capacities
        or not config.projected_widths
        or not config.projected_key_capacities
        or min(config.key_capacities) < 0
        or min(config.candidate_capacities) < 0
        or min(config.projected_widths) <= 0
        or min(config.projected_key_capacities) < 0
        or config.fixed_key_capacity < 0
        or config.fixed_candidate_capacity < 0
    ):
        raise ValueError(
            "memory scaling dimensions must be nonnegative and nonempty"
        )
    rng = np.random.default_rng(config.seed)
    features = rng.normal(size=(config.updates, config.feature_width))
    features /= np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True),
        np.finfo(float).tiny,
    )
    targets = np.eye(config.classes, dtype=np.float64)[
        np.arange(config.updates) % config.classes
    ]
    key_runs = [
        _measure_managed_memory(
            features=features,
            targets=targets,
            key_capacity=capacity,
            candidate_capacity=config.fixed_candidate_capacity,
            classes=config.classes,
            seed=config.seed,
            prefill_keys=True,
        )
        for capacity in config.key_capacities
    ]
    candidate_runs = [
        _measure_managed_memory(
            features=features,
            targets=targets,
            key_capacity=config.fixed_key_capacity,
            candidate_capacity=capacity,
            classes=config.classes,
            seed=config.seed,
            prefill_keys=False,
        )
        for capacity in config.candidate_capacities
    ]
    projected = [
        {
            "feature_width": width,
            "key_capacity": capacity,
            "candidate_capacity": config.fixed_candidate_capacity,
            "projected_state_bytes": _projected_managed_memory_bytes(
                width,
                config.classes,
                capacity,
                config.fixed_candidate_capacity,
            ),
        }
        for width in config.projected_widths
        for capacity in config.projected_key_capacities
    ]
    return {
        "experiment": "managed_memory_capacity_scaling",
        "config": asdict(config),
        "key_capacity_runs": key_runs,
        "candidate_capacity_runs": candidate_runs,
        "projected": projected,
        "generated_stream_storage_bytes": features.nbytes + targets.nbytes,
        "downloaded_data_bytes": 0,
    }


def run_scaling_experiment(
    image_config: BlankImageScalingConfig = BlankImageScalingConfig(),
    feature_config: FeatureWidthScalingConfig = FeatureWidthScalingConfig(),
    memory_config: MemoryCapacityScalingConfig = MemoryCapacityScalingConfig(),
) -> dict[str, Any]:
    return {
        "experiment": "milestone_6_scaling",
        "blank_images": run_blank_image_scaling(image_config),
        "feature_widths": run_feature_width_scaling(feature_config),
        "memory_capacities": run_memory_capacity_scaling(memory_config),
    }

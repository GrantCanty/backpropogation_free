"""Milestone 6 experiments for scalable memory and stability/plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from no_backprop.digits import (
    DigitsProtocol,
    DigitsSplit,
    augment_digits_split,
    build_digits_segments,
    load_digits_split,
)
from no_backprop.experiment import (
    DigitsExperimentConfig,
    DigitsKind,
    _digit_target,
    _evaluate_digits_locked,
    _process_digit_image,
    build_digits_learner,
    run_digits_model,
)


@dataclass(frozen=True)
class Milestone6Config:
    hidden_size: int = 64
    test_per_class: int = 40
    seed: int = 29
    block_size: int = 16
    augmentation_copies: int = 1
    forgetting_factors: tuple[float, ...] = (1.0, 0.9999, 0.999, 0.99, 0.95)
    kinds: tuple[DigitsKind, ...] = (
        "lms",
        "rls",
        "diagonal_rls",
        "block_rls",
        "prototype",
        "protected",
    )


def _digits_config(config: Milestone6Config) -> DigitsExperimentConfig:
    return DigitsExperimentConfig(
        hidden_size=config.hidden_size,
        test_per_class=config.test_per_class,
        seed=config.seed,
        block_size=config.block_size,
        augmentation_copies=config.augmentation_copies,
    )


def run_memory_quality(config: Milestone6Config) -> dict[str, Any]:
    """Compare scalable memories on plain, repeated, augmented, and ordered data."""

    digits_config = _digits_config(config)
    split = load_digits_split(test_per_class=config.test_per_class, seed=config.seed)
    augmented = augment_digits_split(
        split,
        copies=config.augmentation_copies,
        max_shift=digits_config.augmentation_max_shift,
        noise_std=digits_config.augmentation_noise_std,
        seed=config.seed + 3,
    )
    repeated_config = replace(
        digits_config,
        passes=digits_config.passes * (1 + config.augmentation_copies),
    )
    protocol_specs: tuple[
        tuple[DigitsProtocol, DigitsSplit, DigitsExperimentConfig], ...
    ] = (
        ("shuffled", split, digits_config),
        ("shuffled_repeated", split, repeated_config),
        ("shuffled_augmented", augmented, digits_config),
        ("class_ordered", split, digits_config),
    )
    return {
        "protocols": {
            protocol: {
                kind: run_digits_model(
                    kind, protocol, protocol_config, split=protocol_split
                )
                for kind in config.kinds
            }
            for protocol, protocol_split, protocol_config in protocol_specs
        }
    }


def run_forgetting_factor_sweep(config: Milestone6Config) -> dict[str, Any]:
    """Expose exact RLS's stability/plasticity curve on matched streams."""

    base_config = _digits_config(config)
    split = load_digits_split(test_per_class=config.test_per_class, seed=config.seed)
    return {
        "factors": {
            str(factor): {
                "approximate_effective_history": (
                    None if factor == 1.0 else 1.0 / (1.0 - factor)
                ),
                "quality": {
                    protocol: run_digits_model(
                        "rls",
                        protocol,
                        replace(base_config, rls_forgetting_factor=factor),
                        split=split,
                    )
                    for protocol in ("shuffled", "class_ordered")
                },
                "concept_drift": run_drift_model(
                    "rls", config, rls_forgetting_factor=factor
                ),
            }
            for factor in config.forgetting_factors
        }
    }


def _train_phase(
    learner,
    split: DigitsSplit,
    *,
    seed: int,
) -> dict[str, float | int]:
    segments = build_digits_segments(
        split.train_labels, protocol="shuffled", passes=1, seed=seed
    )
    correct = 0
    total = 0
    for segment in segments:
        for index in segment.indices:
            label = int(split.train_labels[index])
            prediction = _process_digit_image(
                learner,
                split.train_images[index],
                target=_digit_target(label),
            )
            correct += int(np.argmax(prediction) == label)
            total += 1
    return {"samples": total, "online_accuracy": correct / total}


def run_drift_model(
    kind: DigitsKind,
    config: Milestone6Config,
    *,
    rls_forgetting_factor: float | None = None,
) -> dict[str, Any]:
    """Train original -> inverted -> original and measure loss and recovery."""

    digits_config = _digits_config(config)
    if rls_forgetting_factor is not None:
        digits_config = replace(
            digits_config, rls_forgetting_factor=rls_forgetting_factor
        )
    original = load_digits_split(
        test_per_class=config.test_per_class, seed=config.seed
    )
    inverted = DigitsSplit(
        train_images=1.0 - original.train_images,
        train_labels=original.train_labels,
        test_images=1.0 - original.test_images,
        test_labels=original.test_labels,
    )
    learner = build_digits_learner(kind, digits_config)
    state_before = learner.state_nbytes
    phases: list[dict[str, Any]] = []
    for phase_index, (name, training_split) in enumerate(
        (("original", original), ("inverted", inverted), ("original_return", original))
    ):
        training = _train_phase(
            learner, training_split, seed=config.seed + 10 + phase_index
        )
        phases.append(
            {
                "phase": name,
                "training": training,
                "original_test": _evaluate_digits_locked(
                    learner, original.test_images, original.test_labels
                ),
                "inverted_test": _evaluate_digits_locked(
                    learner, inverted.test_images, inverted.test_labels
                ),
            }
        )
    original_after_first = phases[0]["original_test"]["accuracy"]
    original_after_drift = phases[1]["original_test"]["accuracy"]
    original_after_return = phases[2]["original_test"]["accuracy"]
    return {
        "model": kind,
        "phases": phases,
        "original_forgetting_after_drift": float(
            original_after_first - original_after_drift
        ),
        "original_recovery_on_return": float(
            original_after_return - original_after_drift
        ),
        "final_original_accuracy": original_after_return,
        "final_inverted_accuracy": phases[-1]["inverted_test"]["accuracy"],
        "state_bytes_before": state_before,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": state_before == learner.state_nbytes,
    }


def run_concept_drift(config: Milestone6Config) -> dict[str, Any]:
    return {
        "protocol": "original_to_inverted_to_original",
        "models": {
            kind: run_drift_model(kind, config) for kind in config.kinds
        },
    }


def run_milestone6(config: Milestone6Config = Milestone6Config()) -> dict[str, Any]:
    return {
        "experiment": "milestone_6_scalable_continual_memory",
        "config": asdict(config),
        "quality": run_memory_quality(config),
        "forgetting_factor_sweep": run_forgetting_factor_sweep(config),
        "concept_drift": run_concept_drift(config),
    }

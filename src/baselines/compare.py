"""Systems comparison between online local learning and matched BPTT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from baselines.bptt import BPTTConfig, run_bptt_signal
from baselines.digits import BPTTDigitsConfig, run_bptt_digits
from no_backprop.digits import augment_digits_split, load_digits_split
from no_backprop.experiment import (
    DigitsExperimentConfig,
    SignalExperimentConfig,
    run_digits_model,
    run_signal_model,
)


@dataclass(frozen=True)
class SystemsComparisonConfig:
    steps: int = 3_000
    regime_length: int = 750
    hidden_size: int = 64
    seed: int = 7
    bptt_windows: tuple[int, ...] = (8, 32, 128)


@dataclass(frozen=True)
class DigitsSystemsComparisonConfig:
    hidden_size: int = 64
    test_per_class: int = 40
    passes: int = 1
    augmentation_copies: int = 1
    augmentation_max_shift: int = 1
    augmentation_noise_std: float = 0.03
    seed: int = 29
    bptt_learning_rate: float = 0.001


def run_systems_comparison(config: SystemsComparisonConfig) -> dict[str, Any]:
    signal_config = SignalExperimentConfig(
        steps=config.steps,
        regime_length=config.regime_length,
        hidden_size=config.hidden_size,
        seed=config.seed,
    )
    online = {
        kind: run_signal_model(kind, signal_config) for kind in ("lms", "rls")
    }
    bptt = {
        str(window): run_bptt_signal(
            BPTTConfig(
                steps=config.steps,
                regime_length=config.regime_length,
                hidden_size=config.hidden_size,
                window=window,
                seed=config.seed,
            )
        )
        for window in config.bptt_windows
    }
    return {
        "experiment": "systems_comparison",
        "config": asdict(config),
        "online": online,
        "bptt": bptt,
    }


def run_digits_systems_comparison(
    config: DigitsSystemsComparisonConfig,
) -> dict[str, Any]:
    """Compare local updates and BPTT on identical shuffled image streams."""

    split = load_digits_split(test_per_class=config.test_per_class, seed=config.seed)
    augmented = augment_digits_split(
        split,
        copies=config.augmentation_copies,
        max_shift=config.augmentation_max_shift,
        noise_std=config.augmentation_noise_std,
        seed=config.seed + 3,
    )
    local_config = DigitsExperimentConfig(
        hidden_size=config.hidden_size,
        test_per_class=config.test_per_class,
        passes=config.passes,
        augmentation_copies=config.augmentation_copies,
        augmentation_max_shift=config.augmentation_max_shift,
        augmentation_noise_std=config.augmentation_noise_std,
        seed=config.seed,
    )
    bptt_config = BPTTDigitsConfig(
        hidden_size=config.hidden_size,
        test_per_class=config.test_per_class,
        passes=config.passes,
        augmentation_copies=config.augmentation_copies,
        augmentation_max_shift=config.augmentation_max_shift,
        augmentation_noise_std=config.augmentation_noise_std,
        seed=config.seed,
        learning_rate=config.bptt_learning_rate,
    )
    repeated_local_config = replace(
        local_config,
        passes=config.passes * (1 + config.augmentation_copies),
    )
    repeated_bptt_config = replace(
        bptt_config,
        passes=config.passes * (1 + config.augmentation_copies),
    )
    kinds = (
        "lms",
        "rls",
        "diagonal_rls",
        "block_rls",
        "prototype",
        "protected",
        "eligibility",
        "fast_slow",
    )
    protocol_specs = (
        ("shuffled", split, local_config, bptt_config),
        (
            "shuffled_repeated",
            split,
            repeated_local_config,
            repeated_bptt_config,
        ),
        ("shuffled_augmented", augmented, local_config, bptt_config),
    )
    return {
        "experiment": "digits_shuffled_systems_comparison",
        "config": asdict(config),
        "dataset": {
            "train_samples": len(split.train_labels),
            "augmented_train_samples": len(augmented.train_labels),
            "test_samples": len(split.test_labels),
        },
        "protocols": {
            protocol: {
                "local": {
                    kind: run_digits_model(
                        kind, protocol, protocol_local_config, split=protocol_split
                    )
                    for kind in kinds
                },
                "bptt": run_bptt_digits(
                    protocol_bptt_config,
                    protocol=protocol,
                    split=protocol_split,
                ),
            }
            for (
                protocol,
                protocol_split,
                protocol_local_config,
                protocol_bptt_config,
            ) in protocol_specs
        },
    }

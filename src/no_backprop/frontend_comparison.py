"""Matched recurrent, pixel, and fixed-convolution frontend experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from no_backprop.digits import (
    DigitsProtocol,
    DigitsSplit,
    augment_digits_split,
    load_digits_split,
)
from no_backprop.experiment import (
    DigitsExperimentConfig,
    DigitsKind,
    run_digits_model,
)
from no_backprop.milestone6 import Milestone6Config, run_drift_model


FRONTEND_KINDS: tuple[DigitsKind, ...] = (
    "probation_managed16",
    "managed16_pixels",
    "managed16_fixed_conv",
)


@dataclass(frozen=True)
class FrontendComparisonConfig:
    """A preregistered 64-feature comparison with one shared memory rule."""

    seeds: tuple[int, ...] = (3, 7, 11, 17, 23, 29, 37, 41, 47, 53)
    test_per_class: int = 40
    augmentation_copies: int = 1
    augmentation_max_shift: int = 1
    augmentation_noise_std: float = 0.03
    protocols: tuple[DigitsProtocol, ...] = (
        "shuffled",
        "shuffled_augmented",
        "class_ordered",
    )
    include_drift: bool = True
    feature_width: int = 64
    cumulative_regularization: float = 1.0
    maturity_max_neurons: int = 32
    maturity_rbf_width: float = 0.05
    maturity_min_center_distance: float = 0.01


def _digits_config(
    config: FrontendComparisonConfig, seed: int
) -> DigitsExperimentConfig:
    if config.feature_width != 64:
        raise ValueError("the matched frontend experiment requires 64 features")
    return DigitsExperimentConfig(
        hidden_size=config.feature_width,
        test_per_class=config.test_per_class,
        augmentation_copies=config.augmentation_copies,
        augmentation_max_shift=config.augmentation_max_shift,
        augmentation_noise_std=config.augmentation_noise_std,
        seed=seed,
        cumulative_regularization=config.cumulative_regularization,
        maturity_max_neurons=config.maturity_max_neurons,
        maturity_rbf_width=config.maturity_rbf_width,
        maturity_min_center_distance=config.maturity_min_center_distance,
    )


def _milestone_config(
    config: FrontendComparisonConfig, seed: int
) -> Milestone6Config:
    return Milestone6Config(
        hidden_size=config.feature_width,
        test_per_class=config.test_per_class,
        seed=seed,
        augmentation_copies=config.augmentation_copies,
        cumulative_regularization=config.cumulative_regularization,
        maturity_max_neurons=config.maturity_max_neurons,
        maturity_rbf_width=config.maturity_rbf_width,
        maturity_min_center_distance=config.maturity_min_center_distance,
        kinds=FRONTEND_KINDS,
    )


def _protocol_splits(
    config: FrontendComparisonConfig,
    seed: int,
) -> tuple[DigitsExperimentConfig, dict[DigitsProtocol, DigitsSplit]]:
    digits_config = _digits_config(config, seed)
    split = load_digits_split(test_per_class=config.test_per_class, seed=seed)
    augmented = augment_digits_split(
        split,
        copies=config.augmentation_copies,
        max_shift=config.augmentation_max_shift,
        noise_std=config.augmentation_noise_std,
        seed=seed + 3,
    )
    return digits_config, {
        "shuffled": split,
        "shuffled_repeated": split,
        "shuffled_augmented": augmented,
        "class_ordered": split,
    }


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": values,
    }


def _quality_value(run: dict[str, Any], metric: str) -> float:
    if metric == "forgetting":
        return float(run["forgetting"]["mean"])
    return float(run[metric])


def _summarize_quality(
    runs: list[dict[str, Any]], protocols: tuple[DigitsProtocol, ...]
) -> dict[str, Any]:
    metrics = (
        "online_accuracy",
        "final_test_accuracy",
        "forgetting",
        "training_images_per_second",
        "state_bytes_after",
    )
    return {
        protocol: {
            kind: {
                metric: _summary(
                    [
                        _quality_value(
                            run["quality"][protocol][kind], metric
                        )
                        for run in runs
                    ]
                )
                for metric in metrics
            }
            for kind in FRONTEND_KINDS
        }
        for protocol in protocols
    }


def _summarize_paired_differences(
    runs: list[dict[str, Any]], protocols: tuple[DigitsProtocol, ...]
) -> dict[str, Any]:
    baseline = "probation_managed16"
    metrics = ("online_accuracy", "final_test_accuracy", "forgetting")
    return {
        protocol: {
            kind: {
                metric: _summary(
                    [
                        _quality_value(
                            run["quality"][protocol][kind], metric
                        )
                        - _quality_value(
                            run["quality"][protocol][baseline], metric
                        )
                        for run in runs
                    ]
                )
                for metric in metrics
            }
            for kind in FRONTEND_KINDS
            if kind != baseline
        }
        for protocol in protocols
    }


def _summarize_drift(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = (
        "original_forgetting_after_drift",
        "original_recovery_on_return",
        "final_original_accuracy",
        "final_inverted_accuracy",
    )
    return {
        kind: {
            metric: _summary(
                [float(run["drift"][kind][metric]) for run in runs]
            )
            for metric in metrics
        }
        for kind in FRONTEND_KINDS
    }


def _summarize_paired_drift_differences(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = "probation_managed16"
    metrics = (
        "original_forgetting_after_drift",
        "original_recovery_on_return",
        "final_original_accuracy",
        "final_inverted_accuracy",
    )
    return {
        kind: {
            metric: _summary(
                [
                    float(run["drift"][kind][metric])
                    - float(run["drift"][baseline][metric])
                    for run in runs
                ]
            )
            for metric in metrics
        }
        for kind in FRONTEND_KINDS
        if kind != baseline
    }


def run_frontend_comparison(
    config: FrontendComparisonConfig = FrontendComparisonConfig(),
) -> dict[str, Any]:
    """Run matched frontends without downloading or retaining raw samples."""

    if not config.seeds or not config.protocols:
        raise ValueError("seeds and protocols cannot be empty")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    allowed_protocols = {"shuffled", "shuffled_augmented", "class_ordered"}
    if any(protocol not in allowed_protocols for protocol in config.protocols):
        raise ValueError(
            "frontend protocols must be shuffled, shuffled_augmented, or "
            "class_ordered"
        )
    runs: list[dict[str, Any]] = []
    for seed in config.seeds:
        digits_config, splits = _protocol_splits(config, seed)
        quality = {
            protocol: {
                kind: run_digits_model(
                    kind,
                    protocol,
                    digits_config,
                    split=splits[protocol],
                )
                for kind in FRONTEND_KINDS
            }
            for protocol in config.protocols
        }
        drift = (
            {
                kind: run_drift_model(
                    kind, _milestone_config(config, seed)
                )
                for kind in FRONTEND_KINDS
            }
            if config.include_drift
            else {}
        )
        runs.append({"seed": seed, "quality": quality, "drift": drift})

    quality_runs = [
        result
        for run in runs
        for models in run["quality"].values()
        for result in models.values()
    ]
    result: dict[str, Any] = {
        "experiment": "matched_nonrecurrent_image_frontends",
        "config": asdict(config),
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "downloaded_data_bytes": 0,
        },
        "invariants": {
            "downstream_memory": "managed 16-candidate probation",
            "same_downstream_memory": all(
                run["maturity_diagnostics"]["managed_candidate_capacity"]
                and run["maturity_diagnostics"]["candidate_capacity"] == 16
                for run in quality_runs
            ),
            "same_readout_feature_width": all(
                run["readout_feature_width"] == config.feature_width
                for run in quality_runs
            ),
            "one_label_update_per_image": all(
                run["maturity_diagnostics"][
                    "samples_in_cumulative_statistics"
                ]
                == run["trained_samples"]
                for run in quality_runs
            ),
            "weights_locked_during_evaluation": all(
                run["weights_locked_during_evaluation"]
                for run in quality_runs
            ),
            "bounded_state": all(run["bounded_state"] for run in quality_runs),
            "raw_samples_stored": 0,
        },
        "runs": runs,
        "summary": {
            "quality": _summarize_quality(runs, config.protocols),
            "paired_difference_from_recurrent": (
                _summarize_paired_differences(runs, config.protocols)
            ),
        },
    }
    if config.include_drift:
        result["summary"]["drift"] = _summarize_drift(runs)
        result["summary"]["paired_drift_difference_from_recurrent"] = (
            _summarize_paired_drift_differences(runs)
        )
    return result

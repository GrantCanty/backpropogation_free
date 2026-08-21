"""Method-neutral recurring image-domain transformations and evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from continual_core.datasets.digits import DigitsSplit, load_digits_split
from experiments.legacy import (
    DigitsExperimentConfig,
    DigitsKind,
    evaluate_digits_locked,
    build_digits_learner,
)
from experiments.milestone6 import train_phase


TransformationName = Literal[
    "inversion",
    "low_contrast",
    "gaussian_noise",
    "center_occlusion",
    "translation",
    "striped_background",
]

PRIMARY_SPATIAL_BASELINE: DigitsKind = "managed16_signed_magnitude_conv"
DRIFT_SUITE_KINDS: tuple[DigitsKind, ...] = (
    PRIMARY_SPATIAL_BASELINE,
    "probation_managed16",
    "managed16_fixed_conv",
    "managed16_absolute_conv",
)


@dataclass(frozen=True)
class DriftSuiteConfig:
    """Matched recurring shifts over the bundled 8x8 digits."""

    seeds: tuple[int, ...] = (3, 7, 11, 17, 23, 29, 37, 41, 47, 53)
    test_per_class: int = 40
    transformations: tuple[TransformationName, ...] = (
        "inversion",
        "low_contrast",
        "gaussian_noise",
        "center_occlusion",
        "translation",
        "striped_background",
    )
    kinds: tuple[DigitsKind, ...] = DRIFT_SUITE_KINDS
    feature_width: int = 64
    cumulative_regularization: float = 1.0
    maturity_max_neurons: int = 32
    maturity_rbf_width: float = 0.05
    maturity_min_center_distance: float = 0.01
    predictor_regularization: float = 1.0
    contrast_scale: float = 0.55
    noise_std: float = 0.16
    occlusion_size: int = 2
    translation_pixels: int = 1
    stripe_strength: float = 0.20

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not self.transformations:
            raise ValueError("transformations cannot be empty")
        if len(set(self.transformations)) != len(self.transformations):
            raise ValueError("transformations must be unique")
        if (
            not self.kinds
            or len(set(self.kinds)) != len(self.kinds)
            or PRIMARY_SPATIAL_BASELINE not in self.kinds
        ):
            raise ValueError(
                "kinds must be unique and include the signed-magnitude baseline"
            )
        allowed = set(TransformationName.__args__)
        if any(name not in allowed for name in self.transformations):
            raise ValueError("unknown drift transformation")
        if self.feature_width != 64:
            raise ValueError("the matched drift suite requires 64 features")
        if self.predictor_regularization <= 0.0:
            raise ValueError("predictor_regularization must be positive")
        if not 0.0 < self.contrast_scale < 1.0:
            raise ValueError("contrast_scale must be in (0, 1)")
        if self.noise_std <= 0.0:
            raise ValueError("noise_std must be positive")
        if not 1 <= self.occlusion_size < 8:
            raise ValueError("occlusion_size must be in [1, 7]")
        if not 1 <= self.translation_pixels < 8:
            raise ValueError("translation_pixels must be in [1, 7]")
        if not 0.0 < self.stripe_strength < 1.0:
            raise ValueError("stripe_strength must be in (0, 1)")


def transform_images(
    images: np.ndarray,
    transformation: TransformationName,
    config: DriftSuiteConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Apply one deterministic, label-preserving 8x8 domain shift."""

    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3 or images.shape[1:] != (8, 8):
        raise ValueError("images must have shape (samples, 8, 8)")
    if transformation == "inversion":
        transformed = 1.0 - images
    elif transformation == "low_contrast":
        transformed = 0.5 + config.contrast_scale * (images - 0.5)
    elif transformation == "gaussian_noise":
        rng = np.random.default_rng(seed)
        transformed = images + rng.normal(0.0, config.noise_std, images.shape)
    elif transformation == "center_occlusion":
        transformed = images.copy()
        start = (8 - config.occlusion_size) // 2
        stop = start + config.occlusion_size
        transformed[:, start:stop, start:stop] = 0.0
    elif transformation == "translation":
        shift = config.translation_pixels
        transformed = np.zeros_like(images)
        transformed[:, shift:, shift:] = images[:, :-shift, :-shift]
    elif transformation == "striped_background":
        stripe = (np.arange(8, dtype=np.float64) % 2.0)[None, None, :]
        transformed = (
            (1.0 - config.stripe_strength) * images
            + config.stripe_strength * stripe
        )
    else:  # pragma: no cover - validated config and Literal guard callers
        raise ValueError(f"unknown transformation: {transformation}")
    return np.clip(transformed, 0.0, 1.0)


def transformed_split(
    split: DigitsSplit,
    transformation: TransformationName,
    config: DriftSuiteConfig,
    *,
    seed: int,
) -> DigitsSplit:
    """Transform train/test images independently while preserving labels."""

    return DigitsSplit(
        train_images=transform_images(
            split.train_images, transformation, config, seed=seed
        ),
        train_labels=split.train_labels.copy(),
        test_images=transform_images(
            split.test_images, transformation, config, seed=seed + 1
        ),
        test_labels=split.test_labels.copy(),
    )


def _digits_config(config: DriftSuiteConfig, seed: int) -> DigitsExperimentConfig:
    return DigitsExperimentConfig(
        hidden_size=config.feature_width,
        test_per_class=config.test_per_class,
        seed=seed,
        cumulative_regularization=config.cumulative_regularization,
        maturity_max_neurons=config.maturity_max_neurons,
        maturity_rbf_width=config.maturity_rbf_width,
        maturity_min_center_distance=config.maturity_min_center_distance,
        predictor_regularization=config.predictor_regularization,
    )


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": values,
    }


DRIFT_METRICS = (
    "zero_shot_transformed_accuracy",
    "transformed_online_accuracy",
    "transformed_accuracy_after_adaptation",
    "original_accuracy_after_adaptation",
    "original_forgetting_after_drift",
    "original_return_online_accuracy",
    "final_original_accuracy",
    "final_transformed_accuracy",
    "original_recovery_on_return",
    "transformed_forgetting_on_return",
    "final_joint_accuracy",
    "training_images_per_second",
    "state_bytes_after",
)


def _run_model_suite(
    kind: DigitsKind,
    config: DriftSuiteConfig,
    original: DigitsSplit,
    transformed: dict[TransformationName, DigitsSplit],
    *,
    seed: int,
) -> dict[str, Any]:
    learner = build_digits_learner(kind, _digits_config(config, seed))
    state_before = learner.state_nbytes
    original_training = train_phase(learner, original, seed=seed + 10)
    original_evaluation = evaluate_digits_locked(
        learner, original.test_images, original.test_labels
    )
    zero_shot = {
        name: evaluate_digits_locked(
            learner, split.test_images, split.test_labels
        )
        for name, split in transformed.items()
    }

    transformations: dict[str, Any] = {}
    for transform_index, (name, shifted) in enumerate(transformed.items()):
        branch = deepcopy(learner)
        started = perf_counter()
        shifted_training = train_phase(
            branch,
            shifted,
            seed=seed + 100 + transform_index,
        )
        original_after_shift = evaluate_digits_locked(
            branch, original.test_images, original.test_labels
        )
        shifted_after_shift = evaluate_digits_locked(
            branch, shifted.test_images, shifted.test_labels
        )
        return_training = train_phase(
            branch,
            original,
            seed=seed + 200 + transform_index,
        )
        final_original = evaluate_digits_locked(
            branch, original.test_images, original.test_labels
        )
        final_shifted = evaluate_digits_locked(
            branch, shifted.test_images, shifted.test_labels
        )
        elapsed = perf_counter() - started
        trained_images = (
            shifted_training["samples"] + return_training["samples"]
        )
        transformations[name] = {
            "zero_shot_transformed_accuracy": zero_shot[name]["accuracy"],
            "transformed_online_accuracy": shifted_training["online_accuracy"],
            "transformed_accuracy_after_adaptation": shifted_after_shift[
                "accuracy"
            ],
            "original_accuracy_after_adaptation": original_after_shift[
                "accuracy"
            ],
            "original_forgetting_after_drift": (
                original_evaluation["accuracy"]
                - original_after_shift["accuracy"]
            ),
            "original_return_online_accuracy": return_training[
                "online_accuracy"
            ],
            "final_original_accuracy": final_original["accuracy"],
            "final_transformed_accuracy": final_shifted["accuracy"],
            "original_recovery_on_return": (
                final_original["accuracy"] - original_after_shift["accuracy"]
            ),
            "transformed_forgetting_on_return": (
                shifted_after_shift["accuracy"] - final_shifted["accuracy"]
            ),
            "final_joint_accuracy": 0.5
            * (final_original["accuracy"] + final_shifted["accuracy"]),
            "training_images_per_second": trained_images / elapsed,
            "state_bytes_after": branch.state_nbytes,
            "bounded_state": state_before == branch.state_nbytes,
            "samples_in_cumulative_statistics": branch.readout.diagnostics[
                "samples_in_cumulative_statistics"
            ],
            "evaluations_locked": all(
                evaluation["weights_unchanged"]
                and evaluation["transient_state_restored"]
                for evaluation in (
                    zero_shot[name],
                    original_after_shift,
                    shifted_after_shift,
                    final_original,
                    final_shifted,
                )
            ),
        }
    return {
        "model": kind,
        "readout_feature_width": learner.readout.input_size - 1,
        "downstream_memory": {
            "mature_capacity": (
                learner.readout.diagnostics["active_neurons"]
                + learner.readout.diagnostics["available_neurons"]
            ),
            "candidate_capacity": learner.readout.diagnostics[
                "candidate_capacity"
            ],
        },
        "state_bytes_before": state_before,
        "state_bytes_after_original": learner.state_nbytes,
        "original_training": original_training,
        "original_accuracy_after_training": original_evaluation["accuracy"],
        "original_evaluation_locked": (
            original_evaluation["weights_unchanged"]
            and original_evaluation["transient_state_restored"]
        ),
        "transformations": transformations,
    }


def _aggregate_model(model: dict[str, Any]) -> dict[str, float]:
    transforms = list(model["transformations"].values())
    final_transformed = [item["final_transformed_accuracy"] for item in transforms]
    return {
        "mean_zero_shot_transformed_accuracy": float(
            np.mean([item["zero_shot_transformed_accuracy"] for item in transforms])
        ),
        "mean_original_forgetting_after_drift": float(
            np.mean([item["original_forgetting_after_drift"] for item in transforms])
        ),
        "mean_final_original_accuracy": float(
            np.mean([item["final_original_accuracy"] for item in transforms])
        ),
        "mean_final_transformed_accuracy": float(np.mean(final_transformed)),
        "worst_final_transformed_accuracy": float(np.min(final_transformed)),
        "mean_final_joint_accuracy": float(
            np.mean([item["final_joint_accuracy"] for item in transforms])
        ),
    }


def run_drift_suite(
    config: DriftSuiteConfig = DriftSuiteConfig(),
) -> dict[str, Any]:
    """Run paired A-to-B-to-A transformations with signed-magnitude baseline."""

    runs: list[dict[str, Any]] = []
    label_preserving = True
    for seed in config.seeds:
        original = load_digits_split(
            test_per_class=config.test_per_class, seed=seed
        )
        transformed = {
            name: transformed_split(
                original,
                name,
                config,
                seed=seed + 1_000 * (index + 1),
            )
            for index, name in enumerate(config.transformations)
        }
        label_preserving = label_preserving and all(
            np.array_equal(split.train_labels, original.train_labels)
            and np.array_equal(split.test_labels, original.test_labels)
            for split in transformed.values()
        )
        models = {
            kind: _run_model_suite(
                kind,
                config,
                original,
                transformed,
                seed=seed,
            )
            for kind in config.kinds
        }
        runs.append({"seed": seed, "models": models})

    transformation_summary = {
        name: {
            kind: {
                metric: _summary(
                    [
                        float(
                            run["models"][kind]["transformations"][name][metric]
                        )
                        for run in runs
                    ]
                )
                for metric in DRIFT_METRICS
            }
            for kind in config.kinds
        }
        for name in config.transformations
    }
    paired = {
        name: {
            kind: {
                metric: _summary(
                    [
                        float(
                            run["models"][kind]["transformations"][name][metric]
                        )
                        - float(
                            run["models"][PRIMARY_SPATIAL_BASELINE][
                                "transformations"
                            ][name][metric]
                        )
                        for run in runs
                    ]
                )
                for metric in DRIFT_METRICS
            }
            for kind in config.kinds
            if kind != PRIMARY_SPATIAL_BASELINE
        }
        for name in config.transformations
    }
    aggregate_runs = [
        {
            kind: _aggregate_model(run["models"][kind])
            for kind in config.kinds
        }
        for run in runs
    ]
    aggregate_metrics = tuple(
        next(iter(aggregate_runs))[PRIMARY_SPATIAL_BASELINE]
    )
    aggregate = {
        kind: {
            metric: _summary(
                [float(run[kind][metric]) for run in aggregate_runs]
            )
            for metric in aggregate_metrics
        }
        for kind in config.kinds
    }
    aggregate_paired = {
        kind: {
            metric: _summary(
                [
                    float(run[kind][metric])
                    - float(run[PRIMARY_SPATIAL_BASELINE][metric])
                    for run in aggregate_runs
                ]
            )
            for metric in aggregate_metrics
        }
        for kind in config.kinds
        if kind != PRIMARY_SPATIAL_BASELINE
    }
    training_samples = len(original.train_labels)
    model_runs = [
        model for run in runs for model in run["models"].values()
    ]
    transformation_runs = [
        transformation
        for model in model_runs
        for transformation in model["transformations"].values()
    ]
    return {
        "experiment": "recurring_transformation_drift_suite",
        "config": asdict(config),
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "downloaded_data_bytes": 0,
            "labels_preserved_by_transformations": label_preserving,
        },
        "baseline": PRIMARY_SPATIAL_BASELINE,
        "controls": [
            kind for kind in config.kinds if kind != PRIMARY_SPATIAL_BASELINE
        ],
        "invariants": {
            "same_downstream_memory": all(
                model["downstream_memory"]
                == {"mature_capacity": 32, "candidate_capacity": 16}
                for model in model_runs
            ),
            "same_readout_feature_width": all(
                model["readout_feature_width"] == config.feature_width
                for model in model_runs
            ),
            "one_label_update_per_image": all(
                item["samples_in_cumulative_statistics"] == 3 * training_samples
                for item in transformation_runs
            ),
            "weights_locked_during_evaluation": all(
                model["original_evaluation_locked"] for model in model_runs
            )
            and all(item["evaluations_locked"] for item in transformation_runs),
            "bounded_state": all(
                item["bounded_state"] for item in transformation_runs
            ),
            "original_phase_trained_once_per_model_and_reused": True,
            "state_reset_before_every_image": True,
            "raw_samples_stored_by_models": 0,
        },
        "runs": runs,
        "summary": {
            "by_transformation": transformation_summary,
            "paired_difference_from_signed_magnitude": paired,
            "aggregate_across_transformations": aggregate,
            "aggregate_paired_difference_from_signed_magnitude": aggregate_paired,
        },
    }

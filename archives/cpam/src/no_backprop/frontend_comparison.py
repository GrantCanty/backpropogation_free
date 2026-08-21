"""Matched recurrent, pixel, and fixed-convolution frontend experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from no_backprop.drift_suite import (
    DriftSuiteConfig,
    TransformationName,
    run_drift_suite,
)
from no_backprop.digits import (
    DigitsProtocol,
    DigitsSplit,
    augment_digits_split,
    load_digits_split,
)
from no_backprop.experiment import (
    DigitsExperimentConfig,
    DigitsKind,
    DigitsLearner,
    build_digits_learner,
    run_digits_model,
)
from no_backprop.milestone6 import Milestone6Config, run_drift_model
from no_backprop.reservoir import OnlineReservoir
from no_backprop.spatial import OnlineSpatialClassifier


FRONTEND_KINDS: tuple[DigitsKind, ...] = (
    "probation_managed16",
    "managed16_pixels",
    "managed16_fixed_conv",
)

PREDICTIVE_KINDS: tuple[DigitsKind, ...] = (
    "probation_managed16",
    "managed16_fixed_conv",
    "managed16_predictive_conv",
)

POLARITY_KINDS: tuple[DigitsKind, ...] = (
    "probation_managed16",
    "managed16_fixed_conv",
    "managed16_absolute_conv",
    "managed16_signed_magnitude_conv",
)

PREDICTIVE_SURPRISE_KINDS: tuple[DigitsKind, ...] = (
    "managed16_signed_magnitude_conv",
    "managed16_predictor_control",
    "managed16_predictive_surprise",
    "managed16_lagged_surprise",
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


@dataclass(frozen=True)
class PredictiveRepresentationConfig(FrontendComparisonConfig):
    """Matched fixed-target latent prediction against both prior controls."""

    predictor_regularization: float = 1.0


@dataclass(frozen=True)
class PolarityComparisonConfig(FrontendComparisonConfig):
    """Matched contrast-polarity geometry ablation."""


@dataclass(frozen=True)
class PredictiveSurpriseComparisonConfig(PredictiveRepresentationConfig):
    """Stable-backbone masked-prediction recruitment experiment."""

    drift_transformations: tuple[TransformationName, ...] = (
        "inversion",
        "low_contrast",
        "gaussian_noise",
        "center_occlusion",
        "translation",
        "striped_background",
    )


ComparisonConfig = (
    FrontendComparisonConfig
    | PredictiveRepresentationConfig
    | PolarityComparisonConfig
    | PredictiveSurpriseComparisonConfig
)


def _digits_config(
    config: ComparisonConfig, seed: int
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
        predictor_regularization=getattr(
            config, "predictor_regularization", 1.0
        ),
    )


def _milestone_config(
    config: ComparisonConfig,
    seed: int,
    kinds: tuple[DigitsKind, ...],
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
        predictor_regularization=getattr(
            config, "predictor_regularization", 1.0
        ),
        kinds=kinds,
    )


def _protocol_splits(
    config: ComparisonConfig,
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
    runs: list[dict[str, Any]],
    protocols: tuple[DigitsProtocol, ...],
    kinds: tuple[DigitsKind, ...],
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
            for kind in kinds
        }
        for protocol in protocols
    }


def _summarize_paired_differences(
    runs: list[dict[str, Any]],
    protocols: tuple[DigitsProtocol, ...],
    kinds: tuple[DigitsKind, ...],
    baseline: DigitsKind,
) -> dict[str, Any]:
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
            for kind in kinds
            if kind != baseline
        }
        for protocol in protocols
    }


def _summarize_drift(
    runs: list[dict[str, Any]], kinds: tuple[DigitsKind, ...]
) -> dict[str, Any]:
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
        for kind in kinds
    }


def _summarize_paired_drift_differences(
    runs: list[dict[str, Any]],
    kinds: tuple[DigitsKind, ...],
    baseline: DigitsKind,
) -> dict[str, Any]:
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
        for kind in kinds
        if kind != baseline
    }


def _run_matched_comparison(
    config: ComparisonConfig,
    *,
    kinds: tuple[DigitsKind, ...],
    baseline: DigitsKind,
    experiment: str,
    paired_summary_name: str,
    paired_drift_summary_name: str,
) -> dict[str, Any]:
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
                for kind in kinds
            }
            for protocol in config.protocols
        }
        drift = (
            {
                kind: run_drift_model(
                    kind, _milestone_config(config, seed, kinds)
                )
                for kind in kinds
            }
            if config.include_drift
            else {}
        )
        runs.append({"seed": seed, "quality": quality, "drift": drift})

    quality_runs = [
        model_result
        for run in runs
        for models in run["quality"].values()
        for model_result in models.values()
    ]
    result: dict[str, Any] = {
        "experiment": experiment,
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
            "quality": _summarize_quality(runs, config.protocols, kinds),
            paired_summary_name: _summarize_paired_differences(
                runs, config.protocols, kinds, baseline
            ),
        },
    }
    if config.include_drift:
        result["summary"]["drift"] = _summarize_drift(runs, kinds)
        result["summary"][paired_drift_summary_name] = (
            _summarize_paired_drift_differences(runs, kinds, baseline)
        )
    return result


def run_frontend_comparison(
    config: FrontendComparisonConfig = FrontendComparisonConfig(),
) -> dict[str, Any]:
    """Run matched frontends without downloading or retaining raw samples."""

    return _run_matched_comparison(
        config,
        kinds=FRONTEND_KINDS,
        baseline="probation_managed16",
        experiment="matched_nonrecurrent_image_frontends",
        paired_summary_name="paired_difference_from_recurrent",
        paired_drift_summary_name="paired_drift_difference_from_recurrent",
    )


def _encode_images_without_learning(
    learner: DigitsLearner, images: np.ndarray
) -> np.ndarray:
    if isinstance(learner, OnlineSpatialClassifier):
        return np.stack([learner.encoder.encode(image) for image in images])
    if not isinstance(learner, OnlineReservoir):
        raise TypeError(
            "polarity geometry requires a fixed spatial or reservoir encoder"
        )
    representations: list[np.ndarray] = []
    no_feedback = np.full(learner.config.output_size, np.nan, dtype=np.float64)
    for image in images:
        learner.reset_state()
        for row in image:
            learner.predict(row)
            learner.learn(no_feedback)
        representations.append(learner.state.copy())
    return np.stack(representations)


def _inversion_geometry(
    kind: DigitsKind,
    config: DigitsExperimentConfig,
    images: np.ndarray,
) -> dict[str, float | int]:
    learner = build_digits_learner(kind, config)
    original = _encode_images_without_learning(learner, images)
    inverted = _encode_images_without_learning(learner, 1.0 - images)
    original_norms = np.linalg.norm(original, axis=1)
    inverted_norms = np.linalg.norm(inverted, axis=1)
    denominator = np.maximum(
        original_norms * inverted_norms, np.finfo(float).tiny
    )
    cosine = np.sum(original * inverted, axis=1) / denominator
    sum_norms = np.maximum(
        original_norms + inverted_norms, np.finfo(float).tiny
    )
    return {
        "samples": len(images),
        "feature_width": original.shape[1],
        "mean_original_inverted_cosine": float(np.mean(cosine)),
        "mean_original_inverted_distance": float(
            np.mean(np.linalg.norm(original - inverted, axis=1))
        ),
        "mean_antipodal_residual": float(
            np.mean(np.linalg.norm(original + inverted, axis=1) / sum_norms)
        ),
    }


def run_polarity_comparison(
    config: PolarityComparisonConfig = PolarityComparisonConfig(),
) -> dict[str, Any]:
    """Test fixed polarity transforms against convolution and recurrence."""

    result = _run_matched_comparison(
        config,
        kinds=POLARITY_KINDS,
        baseline="managed16_fixed_conv",
        experiment="matched_contrast_polarity_frontends",
        paired_summary_name="paired_difference_from_fixed_convolution",
        paired_drift_summary_name=(
            "paired_drift_difference_from_fixed_convolution"
        ),
    )
    for run in result["runs"]:
        seed = int(run["seed"])
        digits_config, splits = _protocol_splits(config, seed)
        images = splits["shuffled"].test_images
        run["inversion_geometry"] = {
            kind: _inversion_geometry(kind, digits_config, images)
            for kind in POLARITY_KINDS
        }
    geometry_metrics = (
        "mean_original_inverted_cosine",
        "mean_original_inverted_distance",
        "mean_antipodal_residual",
    )
    result["summary"]["inversion_geometry"] = {
        kind: {
            metric: _summary(
                [
                    float(run["inversion_geometry"][kind][metric])
                    for run in result["runs"]
                ]
            )
            for metric in geometry_metrics
        }
        for kind in POLARITY_KINDS
    }
    recurrent_kind: DigitsKind = "probation_managed16"
    result["summary"]["absolute_cosine_gap_from_recurrent"] = {
        kind: _summary(
            [
                abs(
                    float(
                        run["inversion_geometry"][kind][
                            "mean_original_inverted_cosine"
                        ]
                    )
                    - float(
                        run["inversion_geometry"][recurrent_kind][
                            "mean_original_inverted_cosine"
                        ]
                    )
                )
                for run in result["runs"]
            ]
        )
        for kind in POLARITY_KINDS
        if kind != recurrent_kind
    }
    result["invariants"].update(
        {
            "geometry_measured_without_learning": True,
            "state_reset_before_every_image": True,
            "polarity_frontends_are_fixed": True,
            "all_geometry_feature_widths_are_64": all(
                geometry["feature_width"] == 64
                for run in result["runs"]
                for geometry in run["inversion_geometry"].values()
            ),
        }
    )
    return result


def run_predictive_representation_comparison(
    config: PredictiveRepresentationConfig = PredictiveRepresentationConfig(),
) -> dict[str, Any]:
    """Compare latent prediction with fixed-convolution and recurrent controls."""

    result = _run_matched_comparison(
        config,
        kinds=PREDICTIVE_KINDS,
        baseline="managed16_fixed_conv",
        experiment="forward_only_predictive_spatial_representation",
        paired_summary_name="paired_difference_from_fixed_convolution",
        paired_drift_summary_name=(
            "paired_drift_difference_from_fixed_convolution"
        ),
    )
    predictive_runs = [
        run["quality"][protocol]["managed16_predictive_conv"]
        for run in result["runs"]
        for protocol in config.protocols
    ]
    result["invariants"].update(
        {
            "predictor_forgetting_factor": 1.0,
            "predictor_uses_backpropagation": False,
            "four_predictor_updates_per_training_image": all(
                run["predictive_diagnostics"]["predictor_updates"]
                == 4 * run["trained_samples"]
                for run in predictive_runs
            ),
            "every_training_image_updates_predictor": all(
                run["predictive_diagnostics"]["predictor_images"]
                == run["trained_samples"]
                for run in predictive_runs
            ),
            "predictor_stored_raw_samples": 0,
        }
    )
    representation_metrics = (
        "target_prediction_mse",
        "effective_rank",
        "normalized_effective_rank",
        "mean_feature_variance",
        "mean_representation_norm",
    )
    result["summary"]["predictive_representation"] = {
        protocol: {
            metric: _summary(
                [
                    float(
                        run["quality"][protocol][
                            "managed16_predictive_conv"
                        ]["representation_diagnostics"][metric]
                    )
                    for run in result["runs"]
                ]
            )
            for metric in representation_metrics
        }
        for protocol in config.protocols
    }
    result["summary"]["predictive_learning"] = {
        protocol: {
            "initial_target_prediction_mse": _summary(
                [
                    float(
                        run["quality"][protocol][
                            "managed16_predictive_conv"
                        ]["evaluation_history"][0][
                            "representation_diagnostics"
                        ]["target_prediction_mse"]
                    )
                    for run in result["runs"]
                ]
            ),
            "final_target_prediction_mse": _summary(
                [
                    float(
                        run["quality"][protocol][
                            "managed16_predictive_conv"
                        ]["representation_diagnostics"][
                            "target_prediction_mse"
                        ]
                    )
                    for run in result["runs"]
                ]
            ),
            "target_prediction_mse_reduction": _summary(
                [
                    float(
                        run["quality"][protocol][
                            "managed16_predictive_conv"
                        ]["evaluation_history"][0][
                            "representation_diagnostics"
                        ]["target_prediction_mse"]
                    )
                    - float(
                        run["quality"][protocol][
                            "managed16_predictive_conv"
                        ]["representation_diagnostics"][
                            "target_prediction_mse"
                        ]
                    )
                    for run in result["runs"]
                ]
            ),
        }
        for protocol in config.protocols
    }
    fixed_metrics = tuple(
        metric
        for metric in representation_metrics
        if metric != "target_prediction_mse"
    )
    result["summary"]["fixed_convolution_representation"] = {
        protocol: {
            metric: _summary(
                [
                    float(
                        run["quality"][protocol]["managed16_fixed_conv"][
                            "representation_diagnostics"
                        ][metric]
                    )
                    for run in result["runs"]
                ]
            )
            for metric in fixed_metrics
        }
        for protocol in config.protocols
    }
    return result


def run_predictive_surprise_comparison(
    config: PredictiveSurpriseComparisonConfig = (
        PredictiveSurpriseComparisonConfig()
    ),
) -> dict[str, Any]:
    """Test whether masked prediction improves bounded recruitment decisions."""

    quality_config = replace(config, include_drift=False)
    result = _run_matched_comparison(
        quality_config,
        kinds=PREDICTIVE_SURPRISE_KINDS,
        baseline="managed16_signed_magnitude_conv",
        experiment="stable_backbone_predictive_surprise_recruitment",
        paired_summary_name="paired_difference_from_signed_magnitude",
        paired_drift_summary_name=(
            "paired_drift_difference_from_signed_magnitude"
        ),
    )
    result["config"] = asdict(config)
    predictive_kinds: tuple[DigitsKind, ...] = (
        "managed16_predictor_control",
        "managed16_predictive_surprise",
        "managed16_lagged_surprise",
    )
    predictive_runs = [
        run["quality"][protocol][kind]
        for run in result["runs"]
        for protocol in config.protocols
        for kind in predictive_kinds
    ]
    result["invariants"].update(
        {
            "classifier_visible_basis_stable": all(
                run["predictive_diagnostics"][
                    "classifier_visible_basis_stable"
                ]
                for run in predictive_runs
            ),
            "predictor_forgetting_factor": 1.0,
            "predictor_uses_backpropagation": False,
            "four_predictor_updates_per_training_image": all(
                run["predictive_diagnostics"]["predictor_updates"]
                == 4 * run["trained_samples"]
                for run in predictive_runs
            ),
            "every_training_image_updates_predictor": all(
                run["predictive_diagnostics"]["predictor_images"]
                == run["trained_samples"]
                for run in predictive_runs
            ),
            "surprise_never_changes_classifier_features": True,
            "surprise_has_no_absolute_threshold": True,
            "lagged_control_stores_only_one_surprise_scalar": True,
            "predictor_stored_raw_samples": 0,
        }
    )
    result["summary"]["predictive_learning"] = {
        protocol: {
            kind: {
                "initial_target_prediction_mse": _summary(
                    [
                        float(
                            run["quality"][protocol][kind][
                                "evaluation_history"
                            ][0]["representation_diagnostics"][
                                "target_prediction_mse"
                            ]
                        )
                        for run in result["runs"]
                    ]
                ),
                "final_target_prediction_mse": _summary(
                    [
                        float(
                            run["quality"][protocol][kind][
                                "representation_diagnostics"
                            ]["target_prediction_mse"]
                        )
                        for run in result["runs"]
                    ]
                ),
            }
            for kind in predictive_kinds
        }
        for protocol in config.protocols
    }
    recruitment_metrics = (
        "active_neurons",
        "candidates_created",
        "candidates_promoted",
        "candidate_pool_rejections",
        "resolved_candidate_reclaims",
        "novelty_candidate_replacements",
    )
    result["summary"]["recruitment"] = {
        protocol: {
            kind: {
                metric: _summary(
                    [
                        float(
                            run["quality"][protocol][kind][
                                "maturity_diagnostics"
                            ][metric]
                        )
                        for run in result["runs"]
                    ]
                )
                for metric in recruitment_metrics
            }
            for kind in PREDICTIVE_SURPRISE_KINDS
        }
        for protocol in config.protocols
    }
    surprise_metrics = (
        "surprise_candidate_assignments",
        "surprise_candidate_replacements",
        "surprise_candidate_rejections",
        "mean_pending_structural_surprise",
    )
    result["summary"]["surprise_recruitment"] = {
        protocol: {
            kind: {
                metric: _summary(
                    [
                        float(
                            run["quality"][protocol][kind][
                                "maturity_diagnostics"
                            ][metric]
                        )
                        for run in result["runs"]
                    ]
                )
                for metric in surprise_metrics
            }
            for kind in (
                "managed16_predictive_surprise",
                "managed16_lagged_surprise",
            )
        }
        for protocol in config.protocols
    }
    if config.include_drift:
        result["drift_suite"] = run_drift_suite(
            DriftSuiteConfig(
                seeds=config.seeds,
                test_per_class=config.test_per_class,
                transformations=config.drift_transformations,
                kinds=PREDICTIVE_SURPRISE_KINDS,
                feature_width=config.feature_width,
                cumulative_regularization=config.cumulative_regularization,
                maturity_max_neurons=config.maturity_max_neurons,
                maturity_rbf_width=config.maturity_rbf_width,
                maturity_min_center_distance=(
                    config.maturity_min_center_distance
                ),
                predictor_regularization=config.predictor_regularization,
            )
        )
    return result

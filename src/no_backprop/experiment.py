"""Reproducible experiment runners and result serialization."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from no_backprop.eligibility import EligibilityConfig, EligibilityReservoir
from no_backprop.digits import (
    DigitsSplit,
    DigitsProtocol,
    augment_digits_split,
    build_digits_segments,
    load_digits_split,
)
from no_backprop.metrics import PrequentialMetrics
from no_backprop.predictive import (
    OnlinePredictiveSpatialClassifier,
    PredictiveSpatialConfig,
)
from no_backprop.readouts import (
    BlockRLSReadout,
    CumulativeMemoryReadout,
    CumulativeMaturityReadout,
    DiagonalRLSReadout,
    FastSlowLMSReadout,
    FastSlowValueProbationaryReadout,
    FrozenReadout,
    KeyValueMaturityReadout,
    LMSReadout,
    ManagedProbationaryMaturityReadout,
    ProbationaryMaturityReadout,
    ProtectedFastSlowReadout,
    PrototypeReadout,
    ResponsibleProbationaryMaturityReadout,
    RLSReadout,
)
from no_backprop.reservoir import OnlineReservoir, ReservoirConfig
from no_backprop.spatial import OnlineSpatialClassifier, SpatialClassifierConfig
from no_backprop.streams import (
    ContinualClassificationConfig,
    DelayedAssociationConfig,
    iter_delayed_association,
    iter_continual_classification,
    iter_nonstationary_signal,
)


ReadoutKind = Literal["frozen", "lms", "rls"]


@dataclass(frozen=True)
class SignalExperimentConfig:
    steps: int = 3_000
    regime_length: int = 750
    hidden_size: int = 64
    seed: int = 7
    window: int = 100
    lms_learning_rate: float = 0.18
    rls_regularization: float = 1.0
    rls_forgetting_factor: float = 0.998


@dataclass(frozen=True)
class DelayedExperimentConfig:
    episodes: int = 1_500
    delay: int = 8
    hidden_size: int = 48
    seed: int = 13
    window: int = 100
    readout_learning_rate: float = 0.12
    trace_decay: float = 0.94
    recurrent_learning_rate: float = 2e-4
    input_learning_rate: float = 1e-4


@dataclass(frozen=True)
class ContinualExperimentConfig:
    steps: int = 4_000
    context_length: int = 1_000
    input_size: int = 8
    classes: int = 3
    hidden_size: int = 48
    seed: int = 17
    window: int = 150
    readout_learning_rate: float = 0.16
    recurrent_learning_rate: float = 1e-4
    trace_decay: float = 0.9
    surprise_threshold: float = 0.75
    fast_decay: float = 0.995
    consolidation_rate: float = 0.002


@dataclass(frozen=True)
class DigitsExperimentConfig:
    """Single-pass continual classification on bundled 8x8 digit images."""

    hidden_size: int = 64
    test_per_class: int = 40
    passes: int = 1
    augmentation_copies: int = 1
    augmentation_max_shift: int = 1
    augmentation_noise_std: float = 0.03
    seed: int = 29
    window: int = 100
    lms_learning_rate: float = 0.18
    rls_regularization: float = 1.0
    rls_forgetting_factor: float = 0.999
    block_size: int = 16
    recurrent_learning_rate: float = 5e-5
    trace_decay: float = 0.88
    surprise_threshold: float = 0.75
    fast_decay: float = 0.995
    consolidation_rate: float = 0.002
    cumulative_regularization: float = 1.0
    cumulative_rank_bins: int = 16
    maturity_max_neurons: int = 32
    maturity_rbf_width: float = 0.05
    maturity_min_center_distance: float = 0.01
    key_prior_strength: float = 4.0
    key_minimum_variance: float = 4e-4
    key_maximum_variance: float = 3.6e-3
    predictor_regularization: float = 1.0


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def build_signal_learner(
    kind: ReadoutKind, config: SignalExperimentConfig
) -> OnlineReservoir:
    reservoir_config = ReservoirConfig(
        input_size=1,
        hidden_size=config.hidden_size,
        output_size=1,
        seed=config.seed,
    )
    feature_size = config.hidden_size + 1
    if kind == "frozen":
        readout = FrozenReadout(feature_size, 1, seed=config.seed)
    elif kind == "lms":
        readout = LMSReadout(
            feature_size,
            1,
            seed=config.seed,
            learning_rate=config.lms_learning_rate,
        )
    elif kind == "rls":
        readout = RLSReadout(
            feature_size,
            1,
            seed=config.seed,
            regularization=config.rls_regularization,
            forgetting_factor=config.rls_forgetting_factor,
        )
    else:
        raise ValueError(f"unknown readout kind: {kind}")
    return OnlineReservoir(reservoir_config, readout)


def run_signal_model(
    kind: ReadoutKind, config: SignalExperimentConfig
) -> dict[str, Any]:
    learner = build_signal_learner(kind, config)
    metrics = PrequentialMetrics()
    initial_state_bytes = learner.state_nbytes
    rss_before = _process_rss_bytes()
    started = time.perf_counter()
    for event in iter_nonstationary_signal(
        config.steps,
        regime_length=config.regime_length,
        seed=config.seed,
    ):
        prediction = learner.predict(event.observation)
        metrics.record(
            prediction,
            event.target,
            regime=event.regime,
            change_point=event.change_point,
        )
        learner.learn(event.target)
    elapsed = time.perf_counter() - started
    rss_after = _process_rss_bytes()
    summary = metrics.summary(config.window)
    summary.update(
        {
            "model": kind,
            "elapsed_seconds": elapsed,
            "steps_per_second": config.steps / elapsed,
            "state_bytes_before": initial_state_bytes,
            "state_bytes_after": learner.state_nbytes,
            "bounded_state": initial_state_bytes == learner.state_nbytes,
            "rss_delta_bytes": (
                None if rss_before is None or rss_after is None else rss_after - rss_before
            ),
            "segments": metrics.segment_summaries(config.window),
            "rolling_mse": metrics.rolling_mse(
                config.window, max(1, config.steps // 200)
            ),
        }
    )
    return summary


def run_signal_experiment(config: SignalExperimentConfig) -> dict[str, Any]:
    models = {
        kind: run_signal_model(kind, config) for kind in ("frozen", "lms", "rls")
    }
    return {
        "experiment": "nonstationary_signal",
        "config": asdict(config),
        "models": models,
    }


def build_delayed_learner(
    *, config: DelayedExperimentConfig, plastic: bool
) -> OnlineReservoir:
    reservoir_config = ReservoirConfig(
        input_size=3,
        hidden_size=config.hidden_size,
        output_size=1,
        spectral_radius=0.88,
        input_scale=0.8,
        leak_rate=0.55,
        seed=config.seed,
    )
    readout = LMSReadout(
        config.hidden_size + 1,
        1,
        seed=config.seed,
        learning_rate=config.readout_learning_rate,
    )
    if not plastic:
        return OnlineReservoir(reservoir_config, readout)
    return EligibilityReservoir(
        reservoir_config,
        readout,
        EligibilityConfig(
            trace_decay=config.trace_decay,
            recurrent_learning_rate=config.recurrent_learning_rate,
            input_learning_rate=config.input_learning_rate,
            seed=config.seed + 1,
        ),
    )


def run_delayed_model(
    *, config: DelayedExperimentConfig, plastic: bool
) -> dict[str, Any]:
    learner = build_delayed_learner(config=config, plastic=plastic)
    squared_errors: list[float] = []
    correct: list[float] = []
    initial_state_bytes = learner.state_nbytes
    started = time.perf_counter()
    stream_config = DelayedAssociationConfig(
        episodes=config.episodes, delay=config.delay, seed=config.seed
    )
    for event in iter_delayed_association(stream_config):
        prediction = learner.predict(event.observation)
        if np.all(np.isfinite(event.target)):
            error = event.target - prediction
            squared_errors.append(float(np.mean(np.square(error))))
            correct.append(float(np.sign(prediction[0]) == np.sign(event.target[0])))
        learner.learn(event.target)
    elapsed = time.perf_counter() - started
    errors = np.asarray(squared_errors, dtype=np.float64)
    accuracies = np.asarray(correct, dtype=np.float64)
    width = min(config.window, len(errors))
    result: dict[str, Any] = {
        "model": "eligibility" if plastic else "fixed",
        "episodes": config.episodes,
        "delay": config.delay,
        "mse": float(np.mean(errors)),
        "head_mse": float(np.mean(errors[:width])),
        "tail_mse": float(np.mean(errors[-width:])),
        "accuracy": float(np.mean(accuracies)),
        "tail_accuracy": float(np.mean(accuracies[-width:])),
        "elapsed_seconds": elapsed,
        "events_per_second": (config.episodes * (config.delay + 2)) / elapsed,
        "state_bytes_before": initial_state_bytes,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": initial_state_bytes == learner.state_nbytes,
    }
    if isinstance(learner, EligibilityReservoir):
        result["diagnostics"] = learner.diagnostics
    return result


def run_delayed_experiment(config: DelayedExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": "delayed_association",
        "config": asdict(config),
        "models": {
            "fixed": run_delayed_model(config=config, plastic=False),
            "eligibility": run_delayed_model(config=config, plastic=True),
        },
    }


ContinualKind = Literal["fixed", "eligibility", "gated", "fast_slow"]


def build_continual_learner(
    kind: ContinualKind, config: ContinualExperimentConfig
) -> OnlineReservoir:
    reservoir_config = ReservoirConfig(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.classes,
        spectral_radius=0.86,
        input_scale=0.7,
        leak_rate=0.7,
        seed=config.seed,
    )
    feature_size = config.hidden_size + 1
    if kind == "fast_slow":
        readout = FastSlowLMSReadout(
            feature_size,
            config.classes,
            seed=config.seed,
            learning_rate=config.readout_learning_rate,
            fast_decay=config.fast_decay,
            consolidation_rate=config.consolidation_rate,
        )
    else:
        readout = LMSReadout(
            feature_size,
            config.classes,
            seed=config.seed,
            learning_rate=config.readout_learning_rate,
        )
    if kind == "fixed":
        return OnlineReservoir(reservoir_config, readout)
    threshold = config.surprise_threshold if kind in ("gated", "fast_slow") else 0.0
    return EligibilityReservoir(
        reservoir_config,
        readout,
        EligibilityConfig(
            trace_decay=config.trace_decay,
            recurrent_learning_rate=config.recurrent_learning_rate,
            input_learning_rate=config.recurrent_learning_rate * 0.5,
            surprise_threshold=threshold,
            seed=config.seed + 1,
        ),
    )


def run_continual_model(
    kind: ContinualKind, config: ContinualExperimentConfig
) -> dict[str, Any]:
    learner = build_continual_learner(kind, config)
    accuracies: list[float] = []
    losses: list[float] = []
    regimes: list[int] = []
    initial_state_bytes = learner.state_nbytes
    started = time.perf_counter()
    stream_config = ContinualClassificationConfig(
        steps=config.steps,
        context_length=config.context_length,
        input_size=config.input_size,
        classes=config.classes,
        seed=config.seed,
    )
    for event in iter_continual_classification(stream_config):
        prediction = learner.predict(event.observation)
        losses.append(float(np.mean(np.square(event.target - prediction))))
        accuracies.append(float(np.argmax(prediction) == np.argmax(event.target)))
        regimes.append(event.regime)
        learner.learn(event.target)
    elapsed = time.perf_counter() - started

    segment_results: list[dict[str, Any]] = []
    starts = list(range(0, config.steps, config.context_length))
    for start in starts:
        stop = min(start + config.context_length, config.steps)
        width = min(config.window, stop - start)
        segment_results.append(
            {
                "context": regimes[start],
                "start": start,
                "stop": stop,
                "head_accuracy": float(np.mean(accuracies[start : start + width])),
                "tail_accuracy": float(np.mean(accuracies[stop - width : stop])),
                "mse": float(np.mean(losses[start:stop])),
            }
        )
    repeated_contexts = [item for item in segment_results if item["context"] == 0]
    retention_delta = (
        repeated_contexts[-1]["head_accuracy"] - repeated_contexts[0]["tail_accuracy"]
        if len(repeated_contexts) > 1
        else 0.0
    )
    result: dict[str, Any] = {
        "model": kind,
        "accuracy": float(np.mean(accuracies)),
        "tail_accuracy": float(np.mean(accuracies[-config.window :])),
        "mse": float(np.mean(losses)),
        "retention_delta": float(retention_delta),
        "segments": segment_results,
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "state_bytes_before": initial_state_bytes,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": initial_state_bytes == learner.state_nbytes,
    }
    if isinstance(learner, EligibilityReservoir):
        result["diagnostics"] = learner.diagnostics
    return result


def run_continual_experiment(config: ContinualExperimentConfig) -> dict[str, Any]:
    return {
        "experiment": "continual_classification",
        "config": asdict(config),
        "models": {
            kind: run_continual_model(kind, config)
            for kind in ("fixed", "eligibility", "gated", "fast_slow")
        },
    }


DigitsKind = Literal[
    "frozen",
    "lms",
    "rls",
    "diagonal_rls",
    "block_rls",
    "prototype",
    "protected",
    "eligibility",
    "fast_slow",
    "cumulative_memory",
    "maturity",
    "maturity_entropy",
    "maturity_leverage",
    "maturity_probation",
    "probation_top4",
    "probation_top2",
    "probation_winner",
    "probation_top4_normalized",
    "probation_top2_normalized",
    "probation_managed32",
    "probation_managed16",
    "probation_managed8",
    "managed16_fast_values",
    "managed16_fast_slow_values",
    "key_value",
    "key_value_entropy",
    "managed16_pixels",
    "managed16_fixed_conv",
    "managed16_absolute_conv",
    "managed16_signed_magnitude_conv",
    "managed16_predictive_conv",
]


DigitsLearner = (
    OnlineReservoir
    | OnlineSpatialClassifier
    | OnlinePredictiveSpatialClassifier
)


def build_digits_learner(
    kind: DigitsKind, config: DigitsExperimentConfig
) -> DigitsLearner:
    """Build a recurrent or full-image classifier for 8x8 images."""

    if kind in (
        "managed16_pixels",
        "managed16_fixed_conv",
        "managed16_absolute_conv",
        "managed16_signed_magnitude_conv",
        "managed16_predictive_conv",
    ):
        if config.hidden_size != 64:
            raise ValueError(
                "matched spatial frontends require hidden_size=64"
            )
        readout = ManagedProbationaryMaturityReadout(
            65,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            max_candidates=16,
        )
        if kind == "managed16_predictive_conv":
            return OnlinePredictiveSpatialClassifier(
                PredictiveSpatialConfig(
                    image_size=8,
                    output_size=10,
                    predictor_regularization=config.predictor_regularization,
                    seed=config.seed,
                ),
                readout,
            )
        frontend = {
            "managed16_pixels": "pixels",
            "managed16_fixed_conv": "fixed_convolution",
            "managed16_absolute_conv": "absolute_convolution",
            "managed16_signed_magnitude_conv": (
                "signed_magnitude_convolution"
            ),
        }[kind]
        return OnlineSpatialClassifier(
            SpatialClassifierConfig(
                image_size=8,
                output_size=10,
                frontend=frontend,
                seed=config.seed,
            ),
            readout,
        )

    reservoir_config = ReservoirConfig(
        input_size=8,
        hidden_size=config.hidden_size,
        output_size=10,
        spectral_radius=0.88,
        input_scale=0.65,
        leak_rate=0.7,
        seed=config.seed,
    )
    feature_size = config.hidden_size + 1
    if kind == "frozen":
        readout = FrozenReadout(feature_size, 10, seed=config.seed)
    elif kind == "rls":
        readout = RLSReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.rls_regularization,
            forgetting_factor=config.rls_forgetting_factor,
        )
    elif kind == "diagonal_rls":
        readout = DiagonalRLSReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.rls_regularization,
            forgetting_factor=config.rls_forgetting_factor,
        )
    elif kind == "block_rls":
        readout = BlockRLSReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.rls_regularization,
            forgetting_factor=config.rls_forgetting_factor,
            block_size=config.block_size,
        )
    elif kind == "prototype":
        readout = PrototypeReadout(feature_size, 10, seed=config.seed)
    elif kind == "protected":
        readout = ProtectedFastSlowReadout(
            feature_size,
            10,
            seed=config.seed,
            fast_learning_rate=config.lms_learning_rate,
            fast_decay=config.fast_decay,
        )
    elif kind == "cumulative_memory":
        readout = CumulativeMemoryReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            rank_bins=config.cumulative_rank_bins,
        )
    elif kind in ("maturity", "maturity_entropy", "maturity_leverage"):
        readout = CumulativeMaturityReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            entropy_gated=kind == "maturity_entropy",
            leverage_gated=kind == "maturity_leverage",
        )
    elif kind == "maturity_probation":
        readout = ProbationaryMaturityReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
        )
    elif kind in (
        "probation_top4",
        "probation_top2",
        "probation_winner",
        "probation_top4_normalized",
        "probation_top2_normalized",
    ):
        responsibility_k = {
            "probation_top4": 4,
            "probation_top2": 2,
            "probation_winner": 1,
            "probation_top4_normalized": 4,
            "probation_top2_normalized": 2,
        }[kind]
        readout = ResponsibleProbationaryMaturityReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            responsibility_k=responsibility_k,
            normalize_responsibility=kind.endswith("_normalized"),
        )
    elif kind in (
        "probation_managed32",
        "probation_managed16",
        "probation_managed8",
    ):
        candidate_capacity = {
            "probation_managed32": 32,
            "probation_managed16": 16,
            "probation_managed8": 8,
        }[kind]
        readout = ManagedProbationaryMaturityReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            max_candidates=candidate_capacity,
        )
    elif kind in ("managed16_fast_values", "managed16_fast_slow_values"):
        readout = FastSlowValueProbationaryReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            max_candidates=16,
            consolidate_values=kind == "managed16_fast_slow_values",
        )
    elif kind in ("key_value", "key_value_entropy"):
        readout = KeyValueMaturityReadout(
            feature_size,
            10,
            seed=config.seed,
            regularization=config.cumulative_regularization,
            max_neurons=config.maturity_max_neurons,
            rbf_width=config.maturity_rbf_width,
            min_center_distance=config.maturity_min_center_distance,
            entropy_gated=kind == "key_value_entropy",
            key_prior_strength=config.key_prior_strength,
            minimum_key_variance=config.key_minimum_variance,
            maximum_key_variance=config.key_maximum_variance,
        )
    elif kind == "fast_slow":
        readout = FastSlowLMSReadout(
            feature_size,
            10,
            seed=config.seed,
            learning_rate=config.lms_learning_rate,
            fast_decay=config.fast_decay,
            consolidation_rate=config.consolidation_rate,
        )
    else:
        readout = LMSReadout(
            feature_size,
            10,
            seed=config.seed,
            learning_rate=config.lms_learning_rate,
        )
    if kind in ("eligibility", "fast_slow"):
        threshold = config.surprise_threshold if kind == "fast_slow" else 0.0
        return EligibilityReservoir(
            reservoir_config,
            readout,
            EligibilityConfig(
                trace_decay=config.trace_decay,
                recurrent_learning_rate=config.recurrent_learning_rate,
                input_learning_rate=config.recurrent_learning_rate * 0.5,
                surprise_threshold=threshold,
                seed=config.seed + 1,
            ),
        )
    return OnlineReservoir(reservoir_config, readout)


def _digit_target(label: int) -> np.ndarray:
    target = np.zeros(10, dtype=np.float64)
    target[label] = 1.0
    return target


def _process_digit_image(
    learner: DigitsLearner,
    image: np.ndarray,
    *,
    target: np.ndarray | None,
) -> np.ndarray:
    """Process one image and optionally learn once from its prediction."""

    learner.reset_state()
    no_feedback = np.full(10, np.nan, dtype=np.float64)
    if isinstance(
        learner, (OnlineSpatialClassifier, OnlinePredictiveSpatialClassifier)
    ):
        prediction = learner.predict(image)
        learner.learn(target if target is not None else no_feedback)
        return prediction

    prediction = np.zeros(10, dtype=np.float64)
    for row_index, row in enumerate(image):
        prediction = learner.predict(row)
        is_last_row = row_index == len(image) - 1
        learner.learn(target if is_last_row and target is not None else no_feedback)
    return prediction


def _evaluate_digits(
    learner: DigitsLearner, images: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    correct = np.zeros(len(labels), dtype=np.float64)
    losses = np.zeros(len(labels), dtype=np.float64)
    for index, (image, label) in enumerate(zip(images, labels)):
        target = _digit_target(int(label))
        prediction = _process_digit_image(learner, image, target=None)
        correct[index] = float(np.argmax(prediction) == label)
        losses[index] = float(np.mean(np.square(target - prediction)))
    per_class = {
        str(class_index): float(np.mean(correct[labels == class_index]))
        for class_index in np.unique(labels)
    }
    return {
        "accuracy": float(np.mean(correct)),
        "mse": float(np.mean(losses)),
        "worst_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def _learner_training_arrays(learner: DigitsLearner) -> tuple[np.ndarray, ...]:
    """Return all arrays that evaluation must never modify."""

    arrays = [learner.input_weights, learner.recurrent_weights, learner.bias]
    if isinstance(learner, OnlinePredictiveSpatialClassifier):
        arrays.extend(
            [
                learner.predictor.weights,
                learner.predictor.inverse_correlation,
                learner.predictor_image_count,
                learner.predictor_update_count,
                learner.predictor_squared_error_sum,
            ]
        )
    if isinstance(learner.readout, CumulativeMaturityReadout):
        arrays.extend(
            [
                learner.readout.expanded_weights,
                learner.readout.inverse_correlation,
                learner.readout.neuron_centers,
                learner.readout.neuron_active,
                learner.readout.neuron_evidence,
                learner.readout.neuron_labels,
                learner.readout.neuron_recruitment_entropy,
                learner.readout.active_count,
                learner.readout.sample_count,
                learner.readout.correct_entropy_sum,
                learner.readout.correct_entropy_count,
                learner.readout.error_count,
                learner.readout.recruitment_candidate_count,
                learner.readout.entropy_rejection_count,
                learner.readout.leverage_rejection_count,
                learner.readout.proximity_rejection_count,
                learner.readout.normalized_leverage_sum,
                learner.readout.normalized_leverage_count,
            ]
        )
        if isinstance(learner.readout, KeyValueMaturityReadout):
            arrays.extend(
                [
                    learner.readout.key_weight,
                    learner.readout.key_m2,
                    learner.readout.key_variance,
                ]
            )
        if isinstance(learner.readout, ProbationaryMaturityReadout):
            arrays.extend(
                [
                    learner.readout.candidate_centers,
                    learner.readout.candidate_counts,
                    learner.readout.candidate_labels,
                    learner.readout.candidate_active,
                    learner.readout.candidates_created,
                    learner.readout.candidates_promoted,
                    learner.readout.candidate_pool_rejections,
                ]
            )
        if isinstance(learner.readout, ResponsibleProbationaryMaturityReadout):
            arrays.extend(
                [
                    learner.readout.responsible_key_sum,
                    learner.readout.responsibility_sample_count,
                ]
            )
        if isinstance(learner.readout, ManagedProbationaryMaturityReadout):
            arrays.extend(
                [
                    learner.readout.candidate_novelty,
                    learner.readout.resolved_candidate_reclaims,
                    learner.readout.novelty_candidate_replacements,
                ]
            )
        if isinstance(learner.readout, FastSlowValueProbationaryReadout):
            arrays.extend(
                [
                    learner.readout.fast_values,
                    learner.readout.fast_inverse_correlation,
                    learner.readout.next_consolidation_evidence,
                    learner.readout.value_consolidation_counts,
                    learner.readout.fast_update_count,
                    learner.readout.maximum_consolidation_shift,
                ]
            )
    elif isinstance(learner.readout, CumulativeMemoryReadout):
        arrays.extend(
            [
                learner.readout.slow_weights,
                learner.readout.slow_inverse_correlation,
                learner.readout.semantic_centroids,
                learner.readout.semantic_counts,
                learner.readout.exception_centroids,
                learner.readout.exception_counts,
                learner.readout.rank_trials,
                learner.readout.rank_correct,
                learner.readout.sample_count,
                learner.readout.rank_update_count,
                learner.readout.selection_counts,
            ]
        )
    elif isinstance(learner.readout, FastSlowLMSReadout):
        arrays.extend(
            [learner.readout.slow_weights, learner.readout.fast_weights]
        )
    elif isinstance(learner.readout, ProtectedFastSlowReadout):
        arrays.extend(
            [learner.readout.slow_memory.centroids, learner.readout.slow_memory.counts]
        )
        arrays.append(learner.readout.fast_weights)
    else:
        arrays.append(learner.readout.weights)
    if isinstance(learner.readout, RLSReadout):
        arrays.append(learner.readout.inverse_correlation)
    elif isinstance(learner.readout, DiagonalRLSReadout):
        arrays.append(learner.readout.inverse_diagonal)
    elif isinstance(learner.readout, BlockRLSReadout):
        arrays.extend(learner.readout.inverse_blocks)
    elif isinstance(learner.readout, PrototypeReadout):
        arrays.append(learner.readout.counts)
    if isinstance(learner, EligibilityReservoir):
        arrays.append(learner.feedback_weights)
    return tuple(arrays)


def _evaluate_digits_locked(
    learner: DigitsLearner, images: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Evaluate without feedback and verify that all learned arrays stay fixed."""

    before = [array.copy() for array in _learner_training_arrays(learner)]
    transient_arrays = [learner.state]
    if isinstance(learner, EligibilityReservoir):
        transient_arrays.extend(
            [learner.recurrent_eligibility, learner.input_eligibility]
        )
    transient_before = [array.copy() for array in transient_arrays]
    try:
        result = _evaluate_digits(learner, images, labels)
        if isinstance(
            learner,
            (OnlineSpatialClassifier, OnlinePredictiveSpatialClassifier),
        ):
            result["representation_diagnostics"] = (
                learner.representation_diagnostics(images)
            )
    finally:
        for previous, current in zip(transient_before, transient_arrays):
            np.copyto(current, previous)
    unchanged = all(
        np.array_equal(previous, current)
        for previous, current in zip(before, _learner_training_arrays(learner))
    )
    if not unchanged:
        raise RuntimeError("evaluation modified no-backprop learner weights")
    result["weights_unchanged"] = True
    result["transient_state_restored"] = True
    return result


def _digits_forgetting(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure loss from each class's best post-exposure score to final score."""

    checkpoints = history[1:]
    final = history[-1]["per_class_accuracy"]
    per_class: dict[str, float] = {}
    for class_key in final:
        class_index = int(class_key)
        eligible = [
            item["per_class_accuracy"][class_key]
            for item in checkpoints
            if class_index in item["classes_seen"]
        ]
        peak = max(eligible) if eligible else final[class_key]
        per_class[class_key] = float(max(0.0, peak - final[class_key]))
    return {
        "mean": float(np.mean(list(per_class.values()))),
        "maximum": float(max(per_class.values())),
        "per_class": per_class,
    }


def run_digits_model(
    kind: DigitsKind,
    protocol: DigitsProtocol,
    config: DigitsExperimentConfig,
    *,
    split: DigitsSplit | None = None,
) -> dict[str, Any]:
    """Train and evaluate one learner on one online image ordering."""

    data = split or load_digits_split(
        test_per_class=config.test_per_class, seed=config.seed
    )
    segments = build_digits_segments(
        data.train_labels,
        protocol=protocol,
        passes=config.passes,
        seed=config.seed + 2,
    )
    learner = build_digits_learner(kind, config)
    initial_state_bytes = learner.state_nbytes
    training_correct: list[float] = []
    training_losses: list[float] = []
    segment_results: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    classes_seen: set[int] = set()
    trained_samples = 0
    training_seconds = 0.0
    evaluation_seconds = 0.0

    started = time.perf_counter()
    initial_evaluation = _evaluate_digits_locked(
        learner, data.test_images, data.test_labels
    )
    evaluation_seconds += time.perf_counter() - started
    evaluation_history.append(
        {
            "trained_samples": 0,
            "pass": -1,
            "segment": -1,
            "focus_class": None,
            "classes_seen": [],
            **initial_evaluation,
        }
    )

    for segment in segments:
        segment_correct: list[float] = []
        segment_losses: list[float] = []
        started = time.perf_counter()
        for index in segment.indices:
            label = int(data.train_labels[index])
            target = _digit_target(label)
            prediction = _process_digit_image(
                learner, data.train_images[index], target=target
            )
            is_correct = float(np.argmax(prediction) == label)
            loss = float(np.mean(np.square(target - prediction)))
            segment_correct.append(is_correct)
            segment_losses.append(loss)
            training_correct.append(is_correct)
            training_losses.append(loss)
            classes_seen.add(label)
            trained_samples += 1
        training_seconds += time.perf_counter() - started

        width = min(config.window, len(segment_correct))
        segment_results.append(
            {
                "pass": segment.pass_index,
                "segment": segment.segment_index,
                "focus_class": segment.focus_class,
                "samples": len(segment.indices),
                "accuracy": float(np.mean(segment_correct)),
                "head_accuracy": float(np.mean(segment_correct[:width])),
                "tail_accuracy": float(np.mean(segment_correct[-width:])),
                "mse": float(np.mean(segment_losses)),
            }
        )
        started = time.perf_counter()
        evaluation = _evaluate_digits_locked(
            learner, data.test_images, data.test_labels
        )
        evaluation_seconds += time.perf_counter() - started
        evaluation_history.append(
            {
                "trained_samples": trained_samples,
                "pass": segment.pass_index,
                "segment": segment.segment_index,
                "focus_class": segment.focus_class,
                "classes_seen": sorted(classes_seen),
                **evaluation,
            }
        )

    final_evaluation = evaluation_history[-1]
    width = min(config.window, len(training_correct))
    result: dict[str, Any] = {
        "model": kind,
        "protocol": protocol,
        "trained_samples": trained_samples,
        "online_accuracy": float(np.mean(training_correct)),
        "tail_online_accuracy": float(np.mean(training_correct[-width:])),
        "online_mse": float(np.mean(training_losses)),
        "initial_test_accuracy": initial_evaluation["accuracy"],
        "final_test_accuracy": final_evaluation["accuracy"],
        "final_test_mse": final_evaluation["mse"],
        "final_worst_class_accuracy": final_evaluation["worst_class_accuracy"],
        "forgetting": _digits_forgetting(evaluation_history),
        "segments": segment_results,
        "evaluation_history": evaluation_history,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "training_images_per_second": trained_samples / training_seconds,
        "readout_feature_width": learner.readout.input_size - 1,
        "state_bytes_before": initial_state_bytes,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": initial_state_bytes == learner.state_nbytes,
        "weights_locked_during_evaluation": all(
            item["weights_unchanged"] for item in evaluation_history
        ),
        "transient_state_restored_after_evaluation": all(
            item["transient_state_restored"] for item in evaluation_history
        ),
    }
    if isinstance(learner, EligibilityReservoir):
        result["diagnostics"] = learner.diagnostics
    if isinstance(learner.readout, CumulativeMemoryReadout):
        result["memory_diagnostics"] = learner.readout.diagnostics
    if isinstance(learner.readout, CumulativeMaturityReadout):
        result["maturity_diagnostics"] = learner.readout.diagnostics
    if isinstance(learner, OnlinePredictiveSpatialClassifier):
        result["predictive_diagnostics"] = learner.diagnostics
    if isinstance(
        learner, (OnlineSpatialClassifier, OnlinePredictiveSpatialClassifier)
    ):
        result["representation_diagnostics"] = final_evaluation[
            "representation_diagnostics"
        ]
    result["frontend_diagnostics"] = (
        learner.diagnostics
        if isinstance(
            learner,
            (OnlineSpatialClassifier, OnlinePredictiveSpatialClassifier),
        )
        else {
            "frontend": "row_reservoir",
            "recurrent": True,
            "image_events_per_prediction": 8,
            "feature_width": config.hidden_size,
            "fixed_frontend": True,
        }
    )
    return result


def run_digits_experiment(config: DigitsExperimentConfig) -> dict[str, Any]:
    """Run matched shuffled and class-ordered 8x8 image benchmarks."""

    split = load_digits_split(test_per_class=config.test_per_class, seed=config.seed)
    augmented_split = augment_digits_split(
        split,
        copies=config.augmentation_copies,
        max_shift=config.augmentation_max_shift,
        noise_std=config.augmentation_noise_std,
        seed=config.seed + 3,
    )
    kinds: tuple[DigitsKind, ...] = (
        "frozen",
        "lms",
        "rls",
        "diagonal_rls",
        "block_rls",
        "prototype",
        "protected",
        "eligibility",
        "fast_slow",
    )
    repeated_config = replace(
        config, passes=config.passes * (1 + config.augmentation_copies)
    )
    protocol_specs = (
        ("shuffled", split, config),
        ("shuffled_repeated", split, repeated_config),
        ("shuffled_augmented", augmented_split, config),
        ("class_ordered", split, config),
    )
    return {
        "experiment": "digits_8x8_continual_classification",
        "config": asdict(config),
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "classes": 10,
            "train_samples": len(split.train_labels),
            "augmented_train_samples": len(augmented_split.train_labels),
            "repeated_train_samples": len(split.train_labels)
            * repeated_config.passes,
            "test_samples": len(split.test_labels),
        },
        "protocols": {
            protocol: {
                kind: run_digits_model(
                    kind,
                    protocol,
                    protocol_config,
                    split=protocol_split,
                )
                for kind in kinds
            }
            for protocol, protocol_split, protocol_config in protocol_specs
        },
    }


def write_json_result(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path

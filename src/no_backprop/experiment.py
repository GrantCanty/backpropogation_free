"""Reproducible experiment runners and result serialization."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from no_backprop.eligibility import EligibilityConfig, EligibilityReservoir
from no_backprop.metrics import PrequentialMetrics
from no_backprop.readouts import (
    FastSlowLMSReadout,
    FrozenReadout,
    LMSReadout,
    RLSReadout,
)
from no_backprop.reservoir import OnlineReservoir, ReservoirConfig
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


def write_json_result(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path

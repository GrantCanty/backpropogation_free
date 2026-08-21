"""Factory-driven online training and locked classification evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import tracemalloc
from typing import Any, Callable, Iterable

import numpy as np

from continual_core.protocols import FloatArray, TaskAdapter
from continual_core.state import locked_state, persistent_state, state_nbytes


TargetEncoder = Callable[[Any], FloatArray]


@dataclass(frozen=True)
class DirectUpdateAdapter:
    """Adapter for learners exposing ``predict`` and ``update`` directly."""

    def predict(self, learner: object, observation: FloatArray) -> FloatArray:
        return learner.predict(observation)  # type: ignore[attr-defined]

    def update(
        self,
        learner: object,
        observation: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
        *,
        learn: bool,
    ) -> None:
        if learn:
            learner.update(  # type: ignore[attr-defined]
                observation, target, prediction
            )


@dataclass(frozen=True)
class FeatureSubsetAdapter:
    """Expose a declared feature subset to an otherwise ordinary readout."""

    indices: tuple[int, ...]

    def _select(self, observation: FloatArray) -> FloatArray:
        return np.asarray(observation, dtype=np.float64)[list(self.indices)]

    def predict(self, learner: object, observation: FloatArray) -> FloatArray:
        return learner.predict(self._select(observation))  # type: ignore[attr-defined]

    def update(
        self,
        learner: object,
        observation: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
        *,
        learn: bool,
    ) -> None:
        if learn:
            learner.update(  # type: ignore[attr-defined]
                self._select(observation), target, prediction
            )


@dataclass(frozen=True)
class PredictLearnAdapter:
    """Adapter for transactional learners exposing ``predict`` then ``learn``."""

    def predict(self, learner: object, observation: FloatArray) -> FloatArray:
        reset = getattr(learner, "reset_state", None)
        if callable(reset):
            reset()
        return learner.predict(observation)  # type: ignore[attr-defined]

    def update(
        self,
        learner: object,
        observation: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
        *,
        learn: bool,
    ) -> None:
        del observation, prediction
        outcome = target if learn else np.full_like(target, np.nan)
        learner.learn(outcome)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RowSequenceAdapter:
    """Adapter for learners that consume image rows as sequential events."""

    output_size: int

    def predict(self, learner: object, observation: FloatArray) -> FloatArray:
        learner.reset_state()  # type: ignore[attr-defined]
        no_feedback = np.full(self.output_size, np.nan, dtype=np.float64)
        prediction = np.zeros(self.output_size, dtype=np.float64)
        for index, row in enumerate(observation):
            prediction = learner.predict(row)  # type: ignore[attr-defined]
            if index < len(observation) - 1:
                learner.learn(no_feedback)  # type: ignore[attr-defined]
        return prediction

    def update(
        self,
        learner: object,
        observation: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
        *,
        learn: bool,
    ) -> None:
        del observation, prediction
        outcome = target if learn else np.full_like(target, np.nan)
        learner.learn(outcome)  # type: ignore[attr-defined]


def train_prequential(
    learner: object,
    events: Iterable[tuple[FloatArray, FloatArray]],
    adapter: TaskAdapter,
) -> dict[str, float | int]:
    """Train once per event and record predictions before each update."""

    correct = 0
    squared_error = 0.0
    samples = 0
    started = perf_counter()
    for observation, target in events:
        prediction = adapter.predict(learner, observation)
        correct += int(np.argmax(prediction) == np.argmax(target))
        squared_error += float(np.mean(np.square(target - prediction)))
        adapter.update(
            learner, observation, target, prediction, learn=True
        )
        samples += 1
    elapsed = perf_counter() - started
    return {
        "samples": samples,
        "accuracy": correct / samples if samples else 0.0,
        "mse": squared_error / samples if samples else 0.0,
        "seconds": elapsed,
        "samples_per_second": samples / elapsed if elapsed else float("inf"),
    }


def evaluate_classification_locked(
    learner: object,
    observations: Iterable[FloatArray],
    labels: Iterable[Any],
    adapter: TaskAdapter,
    encode_target: TargetEncoder,
) -> dict[str, Any]:
    """Evaluate without updates and prove persistent state did not change."""

    correct: list[float] = []
    losses: list[float] = []
    per_label: dict[str, list[float]] = {}
    with locked_state(learner):
        for observation, label in zip(observations, labels):
            target = encode_target(label)
            prediction = adapter.predict(learner, observation)
            hit = float(np.argmax(prediction) == np.argmax(target))
            correct.append(hit)
            losses.append(float(np.mean(np.square(target - prediction))))
            per_label.setdefault(str(label), []).append(hit)
            adapter.update(
                learner, observation, target, prediction, learn=False
            )
    per_label_accuracy = {
        label: float(np.mean(values)) for label, values in per_label.items()
    }
    return {
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "mse": float(np.mean(losses)) if losses else 0.0,
        "worst_class_accuracy": (
            min(per_label_accuracy.values()) if per_label_accuracy else 0.0
        ),
        "per_class_accuracy": per_label_accuracy,
        "weights_unchanged": True,
        "transient_state_restored": True,
        "state_bytes": state_nbytes(learner),
    }


def _latency_summary(durations: list[int]) -> dict[str, float]:
    values = np.asarray(durations, dtype=np.float64) / 1_000.0
    return {
        "mean_microseconds": float(np.mean(values)) if len(values) else 0.0,
        "median_microseconds": float(np.median(values)) if len(values) else 0.0,
        "p95_microseconds": (
            float(np.percentile(values, 95)) if len(values) else 0.0
        ),
    }


def train_classification_profiled(
    learner: object,
    segments: list[list[tuple[FloatArray, FloatArray]]],
    adapter: TaskAdapter,
    evaluation_sets: dict[
        str, tuple[list[FloatArray], list[Any]]
    ],
    encode_target: TargetEncoder,
    *,
    sample_efficiency_steps: Iterable[int] = (),
) -> dict[str, Any]:
    """Profile a segmented prequential stream with locked checkpoints.

    This evaluator is deliberately method-neutral. Learners own updates and
    state; the evaluator owns event order, timing, checkpoints, mutation
    checks, and resource measurement.
    """

    total_events = sum(len(segment) for segment in segments)
    requested_steps = {
        int(step)
        for step in sample_efficiency_steps
        if 0 < int(step) <= total_events
    }
    requested_steps.add(total_events)
    prediction_times: list[int] = []
    update_times: list[int] = []
    correctness: list[float] = []
    squared_errors: list[float] = []
    sample_efficiency: list[dict[str, float | int]] = []
    segment_summaries: list[dict[str, float | int]] = []
    checkpoints: list[dict[str, Any]] = []
    state_before = state_nbytes(learner)

    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    traced_start, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    started = perf_counter()
    samples = 0
    for segment_index, segment in enumerate(segments):
        segment_start = len(correctness)
        for observation, target in segment:
            prediction_started = perf_counter()
            prediction = adapter.predict(learner, observation)
            prediction_times.append(
                int((perf_counter() - prediction_started) * 1_000_000_000)
            )
            correctness.append(
                float(np.argmax(prediction) == np.argmax(target))
            )
            squared_errors.append(float(np.mean(np.square(target - prediction))))
            update_started = perf_counter()
            adapter.update(
                learner, observation, target, prediction, learn=True
            )
            update_times.append(
                int((perf_counter() - update_started) * 1_000_000_000)
            )
            samples += 1
            if samples in requested_steps:
                sample_efficiency.append(
                    {
                        "samples": samples,
                        "cumulative_online_accuracy": float(
                            np.mean(correctness)
                        ),
                        "cumulative_online_mse": float(
                            np.mean(squared_errors)
                        ),
                    }
                )
        segment_values = correctness[segment_start:]
        width = max(1, min(25, len(segment_values) // 5))
        segment_summaries.append(
            {
                "segment": segment_index,
                "samples": len(segment_values),
                "online_accuracy": float(np.mean(segment_values)),
                "head_accuracy": float(np.mean(segment_values[:width])),
                "tail_accuracy": float(np.mean(segment_values[-width:])),
                "adaptation_delta": float(
                    np.mean(segment_values[-width:])
                    - np.mean(segment_values[:width])
                ),
            }
        )
        checkpoint_scores = {
            name: evaluate_classification_locked(
                learner,
                observations,
                labels,
                adapter,
                encode_target,
            )
            for name, (observations, labels) in evaluation_sets.items()
        }
        checkpoints.append(
            {
                "segment": segment_index,
                "samples_seen": samples,
                "evaluation_sets": checkpoint_scores,
            }
        )
    elapsed = perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    if not tracing_before:
        tracemalloc.stop()

    arrays = persistent_state(learner)
    nonfinite = sum(
        int(np.size(array) - np.count_nonzero(np.isfinite(array)))
        for array in arrays.values()
    )
    maximum_absolute_state = max(
        (float(np.max(np.abs(array))) for array in arrays.values() if array.size),
        default=0.0,
    )
    return {
        "samples": samples,
        "online_accuracy": float(np.mean(correctness)) if samples else 0.0,
        "online_mse": float(np.mean(squared_errors)) if samples else 0.0,
        "seconds": elapsed,
        "samples_per_second": samples / elapsed if elapsed else float("inf"),
        "prediction_latency": _latency_summary(prediction_times),
        "update_latency": _latency_summary(update_times),
        "state_bytes_before": state_before,
        "state_bytes_after": state_nbytes(learner),
        "bounded_state": state_before == state_nbytes(learner),
        "peak_traced_training_bytes": max(0, traced_peak - traced_start),
        "peak_process_rss_bytes": None,
        "peak_memory_note": (
            "Python/NumPy traced allocation peak is reported; isolated peak "
            "process RSS is unavailable in this in-process comparison"
        ),
        "nonfinite_state_values": nonfinite,
        "maximum_absolute_state_value": maximum_absolute_state,
        "sample_efficiency": sample_efficiency,
        "segments": segment_summaries,
        "checkpoints": checkpoints,
    }

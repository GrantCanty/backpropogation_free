"""Factory-driven online training and locked classification evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np

from continual_core.protocols import FloatArray, TaskAdapter
from continual_core.state import locked_state, state_nbytes


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


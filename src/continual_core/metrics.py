"""Method-independent prequential metrics for streaming experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def sample_efficiency_steps(
    total_events: int,
    fractions: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
) -> tuple[int, ...]:
    """Return deterministic checkpoints spanning an online stream."""

    if total_events <= 0:
        return tuple()
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise ValueError("sample-efficiency fractions must be in (0, 1]")
    return tuple(
        sorted({max(1, min(total_events, int(total_events * value))) for value in fractions})
    )


def classification_plasticity_summary(
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive stability/plasticity metrics from neutral evaluator output."""

    segments = list(training.get("segments", []))
    checkpoints = list(training.get("checkpoints", []))
    efficiency = list(training.get("sample_efficiency", []))
    total = max(1, int(training.get("samples", 0)))
    if efficiency:
        fractions = np.asarray(
            [float(point["samples"]) / total for point in efficiency],
            dtype=np.float64,
        )
        accuracies = np.asarray(
            [float(point["cumulative_online_accuracy"]) for point in efficiency],
            dtype=np.float64,
        )
        if fractions[0] > 0.0:
            fractions = np.concatenate(([0.0], fractions))
            accuracies = np.concatenate(([accuracies[0]], accuracies))
        sample_auc = float(
            np.sum(
                np.diff(fractions)
                * (accuracies[:-1] + accuracies[1:])
                * 0.5
            )
        )
    else:
        sample_auc = 0.0

    final_all = (
        checkpoints[-1]["evaluation_sets"].get("all", {})
        if checkpoints else {}
    )
    class_names = sorted(
        {
            name
            for checkpoint in checkpoints
            for name in checkpoint["evaluation_sets"]
            if name.startswith("class_")
        },
        key=lambda name: int(name.split("_", 1)[1]),
    )
    class_trajectories = {
        name: np.asarray(
            [
                checkpoint["evaluation_sets"][name]["accuracy"]
                for checkpoint in checkpoints
            ],
            dtype=np.float64,
        )
        for name in class_names
    }
    first_focus: dict[int, int] = {}
    visits: dict[int, list[int]] = {}
    for index, segment in enumerate(segments):
        focus = segment.get("focus_class")
        if focus is not None:
            label = int(focus)
            first_focus.setdefault(label, index)
            visits.setdefault(label, []).append(index)

    forgetting: list[float] = []
    backward_transfer: list[float] = []
    acquisition_accuracy: list[float] = []
    acquisition_gain: list[float] = []
    retention_delta: list[float] = []
    relearning_gain: list[float] = []
    seen: set[int] = set()
    for index, segment in enumerate(segments):
        focus = segment.get("focus_class")
        if focus is None or index >= len(checkpoints):
            continue
        label = int(focus)
        key = f"class_{label}"
        if key not in class_trajectories:
            continue
        trajectory = class_trajectories[key]
        before = float(trajectory[index - 1]) if index else 0.0
        after = float(trajectory[index])
        if label in seen:
            relearning_gain.append(after - before)
        else:
            acquisition_accuracy.append(after)
            acquisition_gain.append(after - before)
        old_labels = sorted(seen - {label})
        if old_labels and index:
            changes = [
                float(class_trajectories[f"class_{old}"][index])
                - float(class_trajectories[f"class_{old}"][index - 1])
                for old in old_labels
                if f"class_{old}" in class_trajectories
            ]
            if changes:
                retention_delta.append(float(np.mean(changes)))
        seen.add(label)

    for label, first_index in first_focus.items():
        key = f"class_{label}"
        if key not in class_trajectories:
            continue
        trajectory = class_trajectories[key]
        learned = trajectory[first_index:]
        if len(learned):
            forgetting.append(float(np.max(learned) - learned[-1]))
            backward_transfer.append(float(learned[-1] - learned[0]))

    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        "sample_efficiency_auc": sample_auc,
        "mean_segment_adaptation_delta": mean(
            [float(segment["adaptation_delta"]) for segment in segments]
        ),
        "mean_events_to_90pct_tail_accuracy": mean(
            [float(segment["events_to_90pct_tail_accuracy"]) for segment in segments]
        ),
        "mean_new_class_acquisition_accuracy": mean(acquisition_accuracy),
        "mean_new_class_acquisition_gain": mean(acquisition_gain),
        "mean_old_class_retention_delta": mean(retention_delta),
        "average_forgetting": mean(forgetting),
        "backward_transfer": mean(backward_transfer),
        "mean_relearning_gain": mean(relearning_gain),
        "focused_segment_count": sum(
            segment.get("focus_class") is not None for segment in segments
        ),
        "new_class_count": len(acquisition_accuracy),
        "recurring_visit_count": sum(max(0, len(items) - 1) for items in visits.values()),
        "final_locked_accuracy": float(final_all.get("accuracy", 0.0)),
        "final_worst_class_accuracy": float(final_all.get("worst_class_accuracy", 0.0)),
    }


@dataclass
class PrequentialMetrics:
    squared_errors: list[float] = field(default_factory=list)
    regimes: list[int] = field(default_factory=list)
    change_points: list[int] = field(default_factory=list)

    def record(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        regime: int,
        change_point: bool,
    ) -> None:
        if np.all(np.isfinite(target)):
            error = np.asarray(target) - np.asarray(prediction)
            self.squared_errors.append(float(np.mean(np.square(error))))
            self.regimes.append(regime)
            if change_point:
                self.change_points.append(len(self.squared_errors) - 1)

    def summary(self, window: int = 100) -> dict[str, float | int | list[int]]:
        if not self.squared_errors:
            raise ValueError("no scored events were recorded")
        errors = np.asarray(self.squared_errors, dtype=np.float64)
        width = min(window, len(errors))
        head = float(np.mean(errors[:width]))
        tail = float(np.mean(errors[-width:]))
        return {
            "scored_steps": len(errors),
            "mse": float(np.mean(errors)),
            "head_mse": head,
            "tail_mse": tail,
            "improvement_ratio": head / max(tail, np.finfo(float).tiny),
            "change_points": self.change_points,
        }

    def segment_summaries(self, window: int = 100) -> list[dict[str, float | int]]:
        errors = np.asarray(self.squared_errors, dtype=np.float64)
        regimes = np.asarray(self.regimes, dtype=np.int64)
        if len(errors) == 0:
            return []
        starts = [0]
        starts.extend(index for index in range(1, len(regimes)) if regimes[index] != regimes[index - 1])
        starts.append(len(errors))
        summaries: list[dict[str, float | int]] = []
        for start, stop in zip(starts, starts[1:]):
            width = min(window, stop - start)
            summaries.append(
                {
                    "regime": int(regimes[start]),
                    "start": start,
                    "stop": stop,
                    "head_mse": float(np.mean(errors[start : start + width])),
                    "tail_mse": float(np.mean(errors[stop - width : stop])),
                }
            )
        return summaries

    def rolling_mse(
        self, window: int = 100, stride: int = 10
    ) -> list[dict[str, float | int]]:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        errors = np.asarray(self.squared_errors, dtype=np.float64)
        if len(errors) == 0:
            return []
        points: list[dict[str, float | int]] = []
        for stop in range(1, len(errors) + 1, stride):
            start = max(0, stop - window)
            points.append({"step": stop, "mse": float(np.mean(errors[start:stop]))})
        if points[-1]["step"] != len(errors):
            start = max(0, len(errors) - window)
            points.append(
                {"step": len(errors), "mse": float(np.mean(errors[start:]))}
            )
        return points

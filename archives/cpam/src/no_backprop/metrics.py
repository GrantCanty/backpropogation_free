"""Prequential metrics for streaming experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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

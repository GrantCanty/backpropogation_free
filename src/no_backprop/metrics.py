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

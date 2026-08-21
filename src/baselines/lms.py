"""Least-mean-squares online baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.frozen import FrozenReadout
from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class LMSReadout(FrozenReadout):
    learning_rate: float = 0.2
    normalized: bool = True
    epsilon: float = 1e-6
    update_clip: float | None = 1.0

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        prediction = vector("prediction", prediction, self.output_size)
        scale = self.learning_rate
        if self.normalized:
            scale /= self.epsilon + float(features @ features)
        update = scale * np.outer(target - prediction, features)
        if self.update_clip is not None:
            update = np.clip(update, -self.update_clip, self.update_clip)
        self.weights += update

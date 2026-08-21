"""Local eligibility traces and recurrent plasticity without a backward pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from no_backprop.protocol import FloatArray, ProtocolError
from no_backprop.readouts import Readout
from no_backprop.reservoir import OnlineReservoir, ReservoirConfig


@dataclass(frozen=True)
class EligibilityConfig:
    trace_decay: float = 0.94
    recurrent_learning_rate: float = 2e-4
    input_learning_rate: float = 1e-4
    update_clip: float = 0.01
    weight_decay: float = 1e-6
    recurrent_row_norm_limit: float = 2.0
    feedback_scale: float = 0.5
    surprise_threshold: float = 0.0
    seed: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.trace_decay < 1.0:
            raise ValueError("trace_decay must be in [0, 1)")
        if min(self.recurrent_learning_rate, self.input_learning_rate) < 0.0:
            raise ValueError("learning rates cannot be negative")
        if self.update_clip <= 0.0:
            raise ValueError("update_clip must be positive")
        if self.recurrent_row_norm_limit <= 0.0:
            raise ValueError("recurrent_row_norm_limit must be positive")
        if self.surprise_threshold < 0.0:
            raise ValueError("surprise_threshold cannot be negative")


class EligibilityReservoir(OnlineReservoir):
    """Reservoir whose input and recurrent weights learn from local traces.

    Each synapse retains a decaying trace of pre/post activity. A fixed random
    projection broadcasts output error to hidden units. No forward weight is
    transported and no historical activation is retained.
    """

    def __init__(
        self,
        config: ReservoirConfig,
        readout: Readout,
        eligibility: EligibilityConfig = EligibilityConfig(),
    ) -> None:
        super().__init__(config, readout)
        self.eligibility_config = eligibility
        self.recurrent_eligibility = np.zeros_like(self.recurrent_weights)
        self.input_eligibility = np.zeros_like(self.input_weights)
        rng = np.random.default_rng(eligibility.seed)
        self.feedback_weights = rng.normal(
            0.0,
            eligibility.feedback_scale / np.sqrt(config.output_size),
            size=(config.hidden_size, config.output_size),
        )
        self._update_count = 0
        self._last_recurrent_update_norm = 0.0
        self._last_input_update_norm = 0.0
        self._last_plasticity_gate = 1.0

    def predict(self, observation: FloatArray) -> FloatArray:
        if self._pending_prediction is not None:
            raise ProtocolError("learn must be called before the next prediction")
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.config.input_size,):
            raise ValueError(
                f"observation must have shape {(self.config.input_size,)}"
            )
        previous_state = self.state.copy()
        preactivation = (
            self.input_weights @ observation
            + self.recurrent_weights @ previous_state
            + self.bias
        )
        candidate = np.tanh(preactivation)
        leak = self.config.leak_rate
        self.state = (1.0 - leak) * previous_state + leak * candidate

        local_sensitivity = leak * (1.0 - np.square(candidate))
        trace_decay = self.eligibility_config.trace_decay
        self.recurrent_eligibility *= trace_decay
        self.recurrent_eligibility += np.outer(local_sensitivity, previous_state)
        self.input_eligibility *= trace_decay
        self.input_eligibility += np.outer(local_sensitivity, observation)

        features = np.concatenate((self.state, np.ones(1, dtype=np.float64)))
        prediction = self.readout.predict(features)
        self._pending_features = features
        self._pending_prediction = prediction.copy()
        return prediction.copy()

    def learn(self, target: FloatArray) -> FloatArray:
        if self._pending_prediction is None or self._pending_features is None:
            raise ProtocolError("predict must be called before learn")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (self.config.output_size,):
            raise ValueError(f"target must have shape {(self.config.output_size,)}")
        prediction = self._pending_prediction
        error = target - prediction
        if np.all(np.isfinite(target)):
            error_norm = float(np.linalg.norm(error))
            threshold = self.eligibility_config.surprise_threshold
            gate = 1.0 if threshold == 0.0 else min(1.0, error_norm / threshold)
            hidden_signal = gate * (self.feedback_weights @ error)
            recurrent_update = (
                self.eligibility_config.recurrent_learning_rate
                * hidden_signal[:, None]
                * self.recurrent_eligibility
            )
            input_update = (
                self.eligibility_config.input_learning_rate
                * hidden_signal[:, None]
                * self.input_eligibility
            )
            clip = self.eligibility_config.update_clip
            np.clip(recurrent_update, -clip, clip, out=recurrent_update)
            np.clip(input_update, -clip, clip, out=input_update)
            self.recurrent_weights *= 1.0 - self.eligibility_config.weight_decay
            self.recurrent_weights += recurrent_update
            self.input_weights += input_update
            self._constrain_recurrent_rows()
            self._last_recurrent_update_norm = float(np.linalg.norm(recurrent_update))
            self._last_input_update_norm = float(np.linalg.norm(input_update))
            self._last_plasticity_gate = gate
            self._update_count += 1
            self.readout.update(self._pending_features, target, prediction)
        self._pending_features = None
        self._pending_prediction = None
        return error.copy()

    def _constrain_recurrent_rows(self) -> None:
        limit = self.eligibility_config.recurrent_row_norm_limit
        norms = np.linalg.norm(self.recurrent_weights, axis=1, keepdims=True)
        scales = np.minimum(1.0, limit / np.maximum(norms, np.finfo(float).tiny))
        self.recurrent_weights *= scales

    def reset_state(self) -> None:
        """Reset transient activity and traces at an observable sequence boundary."""

        super().reset_state()
        self.recurrent_eligibility.fill(0.0)
        self.input_eligibility.fill(0.0)

    @property
    def diagnostics(self) -> dict[str, float | int]:
        return {
            "updates": self._update_count,
            "recurrent_update_norm": self._last_recurrent_update_norm,
            "input_update_norm": self._last_input_update_norm,
            "eligibility_norm": float(np.linalg.norm(self.recurrent_eligibility)),
            "state_saturation": float(np.mean(np.abs(self.state) > 0.95)),
            "plasticity_gate": self._last_plasticity_gate,
        }

    @property
    def state_nbytes(self) -> int:
        return (
            super().state_nbytes
            + self.recurrent_eligibility.nbytes
            + self.input_eligibility.nbytes
            + self.feedback_weights.nbytes
        )

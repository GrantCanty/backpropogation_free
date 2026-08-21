"""A continuously active recurrent reservoir with an online readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from no_backprop.protocol import FloatArray, ProtocolError
from no_backprop.readouts import Readout


@dataclass(frozen=True)
class ReservoirConfig:
    input_size: int = 1
    hidden_size: int = 64
    output_size: int = 1
    spectral_radius: float = 0.92
    input_scale: float = 0.7
    bias_scale: float = 0.05
    leak_rate: float = 0.35
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.input_size, self.hidden_size, self.output_size) <= 0:
            raise ValueError("network dimensions must be positive")
        if not 0.0 < self.leak_rate <= 1.0:
            raise ValueError("leak_rate must be in (0, 1]")
        if self.spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be positive")


class OnlineReservoir:
    """Stateful predict-before-learn reservoir.

    The recurrent dynamics never stop. Only the readout changes in the initial
    vertical slice, keeping learning local and bounded in memory.
    """

    def __init__(self, config: ReservoirConfig, readout: Readout) -> None:
        expected_features = config.hidden_size + 1
        if readout.input_size != expected_features:
            raise ValueError(
                f"readout input_size must be hidden_size + 1 ({expected_features})"
            )
        if readout.output_size != config.output_size:
            raise ValueError("readout output_size does not match reservoir")
        self.config = config
        self.readout = readout
        rng = np.random.default_rng(config.seed)
        self.input_weights = rng.uniform(
            -config.input_scale,
            config.input_scale,
            size=(config.hidden_size, config.input_size),
        )
        recurrent = rng.normal(
            0.0,
            1.0 / np.sqrt(config.hidden_size),
            size=(config.hidden_size, config.hidden_size),
        )
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self.recurrent_weights = recurrent * (config.spectral_radius / radius)
        self.bias = rng.uniform(
            -config.bias_scale, config.bias_scale, size=config.hidden_size
        )
        self.state = np.zeros(config.hidden_size, dtype=np.float64)
        self._pending_features: FloatArray | None = None
        self._pending_prediction: FloatArray | None = None

    def predict(self, observation: FloatArray) -> FloatArray:
        if self._pending_prediction is not None:
            raise ProtocolError("learn must be called before the next prediction")
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (self.config.input_size,):
            raise ValueError(
                f"observation must have shape {(self.config.input_size,)}"
            )
        candidate = np.tanh(
            self.input_weights @ observation
            + self.recurrent_weights @ self.state
            + self.bias
        )
        leak = self.config.leak_rate
        self.state = (1.0 - leak) * self.state + leak * candidate
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
            self.readout.update(self._pending_features, target, prediction)
        self._pending_features = None
        self._pending_prediction = None
        return error.copy()

    def reset_state(self) -> None:
        if self._pending_prediction is not None:
            raise ProtocolError("cannot reset between predict and learn")
        self.state.fill(0.0)

    @property
    def state_nbytes(self) -> int:
        arrays = (
            self.input_weights,
            self.recurrent_weights,
            self.bias,
            self.state,
        )
        return sum(array.nbytes for array in arrays) + self.readout.state_nbytes

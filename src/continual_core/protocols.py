"""Public contracts for method-independent online learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StreamEvent:
    """One prequential event from a streaming environment."""

    step: int
    observation: FloatArray
    target: FloatArray
    regime: int
    change_point: bool = False


class ProtocolError(RuntimeError):
    """Raised when predict and learn are called out of order."""


@runtime_checkable
class OnlineReadout(Protocol):
    """A feature-vector learner with predict-before-update semantics."""

    input_size: int
    output_size: int

    def predict(self, features: FloatArray) -> FloatArray: ...

    def update(
        self,
        features: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
    ) -> None: ...

    @property
    def persistent_arrays(self) -> tuple[NDArray[Any], ...]: ...

    @property
    def state_nbytes(self) -> int: ...

    @property
    def diagnostics(self) -> Mapping[str, object]: ...


class ReadoutFactory(Protocol):
    """Construct an injected readout without coupling a method to a baseline."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        *,
        seed: int,
        regularization: float,
    ) -> OnlineReadout: ...


@runtime_checkable
class StatefulLearner(Protocol):
    """Public state contract used by evaluation and checkpointing."""

    @property
    def persistent_state(self) -> Mapping[str, NDArray[Any]]: ...

    @property
    def transient_state(self) -> Mapping[str, NDArray[Any]]: ...

    @property
    def state_nbytes(self) -> int: ...


@runtime_checkable
class TaskAdapter(Protocol):
    """Translate task observations into method-independent learner calls."""

    def predict(self, learner: object, observation: FloatArray) -> FloatArray: ...

    def update(
        self,
        learner: object,
        observation: FloatArray,
        target: FloatArray,
        prediction: FloatArray,
        *,
        learn: bool,
    ) -> None: ...

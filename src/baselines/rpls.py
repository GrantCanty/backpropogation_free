"""Factor-free cumulative recursive partial least-squares baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.frozen import FrozenReadout
from continual_core.protocols import FloatArray
from continual_core.validation import vector


@dataclass
class RecursivePLSReadout(FrozenReadout):
    """Cumulative PLS regression from recursively updated cross-products.

    For each output, PLS is represented as a ridge solve in the Krylov space
    ``span(c, Sc, ..., S^(components-1)c)``, where ``S = X.T @ X`` and
    ``c = X.T @ y``.  Updating ``S`` and ``c`` is recursive and stores no raw
    observations.  This is an explicit factor-free operational definition of
    RPLS; it is not an exponentially weighted or moving-window variant.
    """

    components: int = 8
    regularization: float = 1.0
    orthogonality_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.components <= 0:
            raise ValueError("components must be positive")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.orthogonality_tolerance <= 0.0:
            raise ValueError("orthogonality_tolerance must be positive")
        self.covariance = np.zeros(
            (self.input_size, self.input_size), dtype=np.float64
        )
        self.cross_covariance = np.zeros(
            (self.input_size, self.output_size), dtype=np.float64
        )
        self.sample_count = np.zeros(1, dtype=np.int64)
        self.effective_components = np.zeros(self.output_size, dtype=np.int64)

    def _fit_output(self, cross: FloatArray) -> tuple[FloatArray, int]:
        basis: list[FloatArray] = []
        direction = cross.copy()
        limit = min(self.components, self.input_size)
        for _ in range(limit):
            for existing in basis:
                direction -= existing * float(existing @ direction)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm <= self.orthogonality_tolerance:
                break
            unit = direction / norm
            basis.append(unit)
            direction = self.covariance @ unit
        if not basis:
            return np.zeros(self.input_size, dtype=np.float64), 0
        projection = np.column_stack(basis)
        reduced_covariance = projection.T @ self.covariance @ projection
        reduced_covariance.flat[:: len(basis) + 1] += self.regularization
        reduced_cross = projection.T @ cross
        try:
            coefficients = np.linalg.solve(reduced_covariance, reduced_cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(
                reduced_covariance, reduced_cross, rcond=None
            )[0]
        return projection @ coefficients, len(basis)

    def update(
        self, features: FloatArray, target: FloatArray, prediction: FloatArray
    ) -> None:
        features = vector("features", features, self.input_size)
        target = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        self.covariance += np.outer(features, features)
        self.cross_covariance += np.outer(features, target)
        for output in range(self.output_size):
            coefficients, count = self._fit_output(
                self.cross_covariance[:, output]
            )
            self.weights[output] = coefficients
            self.effective_components[output] = count
        self.sample_count[0] += 1

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.weights,
            self.covariance,
            self.cross_covariance,
            self.sample_count,
            self.effective_components,
        )

    @property
    def persistent_state(self) -> dict[str, np.ndarray]:
        return {
            "weights": self.weights,
            "covariance": self.covariance,
            "cross_covariance": self.cross_covariance,
            "sample_count": self.sample_count,
            "effective_components": self.effective_components,
        }

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "cumulative_krylov_rpls",
            "components": self.components,
            "effective_components": self.effective_components.tolist(),
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "forgetting_factor": None,
            "stored_raw_observations": 0,
        }

"""Diagonal-corrected cumulative Nyström covariance readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from continual_core.protocols import FloatArray
from continual_core.validation import positive, vector


@dataclass
class NystromCovarianceReadout:
    """Factor-free cumulative ridge regression using fixed Gaussian probes."""

    input_size: int
    output_size: int
    rank: int = 16
    seed: int = 0
    regularization: float = 1.0
    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("all dimensions must be positive")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        positive("regularization", self.regularization)
        positive("epsilon", self.epsilon)
        rng = np.random.default_rng(self.seed)
        self.probes = rng.normal(size=(self.input_size, self.rank))
        self.feature_diagonal = np.zeros(self.input_size, dtype=np.float64)
        self.range_statistic = np.zeros((self.input_size, self.rank), dtype=np.float64)
        self.probe_covariance = np.zeros((self.rank, self.rank), dtype=np.float64)
        self.cross_target = np.zeros((self.input_size, self.output_size), dtype=np.float64)
        self.weights = np.zeros((self.output_size, self.input_size), dtype=np.float64)
        self.sample_count = np.zeros(1, dtype=np.float64)

    def predict(self, features: FloatArray) -> FloatArray:
        values = vector("features", features, self.input_size)
        return self.weights @ values

    # Short mathematical names are public inspection hooks for experiments.
    @property
    def v(self) -> np.ndarray:
        return self.feature_diagonal

    @property
    def Omega(self) -> np.ndarray:
        return self.probes

    @property
    def Y(self) -> np.ndarray:
        return self.range_statistic

    @property
    def G(self) -> np.ndarray:
        return self.probe_covariance

    @property
    def C(self) -> np.ndarray:
        return self.cross_target

    def approximate_covariance(self) -> np.ndarray:
        """Materialize the current approximation for diagnostics only.

        The returned matrix is transient and is never retained in persistent
        state; the online update path uses the structured Woodbury solve.
        """
        k = self.probe_covariance + self.epsilon * np.eye(self.rank)
        low_rank = self.range_statistic @ self._stable_solve(k, self.range_statistic.T)
        diagonal = np.maximum(self.feature_diagonal - np.diag(low_rank), 0.0)
        covariance = low_rank
        covariance[np.diag_indices(self.input_size)] += diagonal + self.regularization
        return 0.5 * (covariance + covariance.T)

    def _solve_weights(self) -> None:
        # A = D + Y K^-1 Y.T, K = G + eps I.  Woodbury avoids a d x d solve.
        k = self.probe_covariance + self.epsilon * np.eye(self.rank)
        k_inverse_y_t = self._stable_solve(k, self.range_statistic.T)
        diagonal_nystrom = np.sum(self.range_statistic * k_inverse_y_t.T, axis=1)
        diagonal = np.maximum(self.feature_diagonal - diagonal_nystrom, 0.0)
        diagonal += self.regularization
        inv_d = 1.0 / diagonal
        d_inv_y = inv_d[:, None] * self.range_statistic
        small = k + self.range_statistic.T @ d_inv_y
        rhs = inv_d[:, None] * self.cross_target
        correction = d_inv_y @ self._stable_solve(
            small, self.range_statistic.T @ rhs
        )
        solved = rhs - correction
        self.weights[...] = solved.T

    @staticmethod
    def _stable_solve(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve a theoretically positive-definite system robustly at startup."""
        matrix = 0.5 * (matrix + matrix.T)
        scale = max(1.0, float(np.max(np.abs(np.diag(matrix)))))
        jitter = np.finfo(np.float64).eps * scale
        for _ in range(7):
            try:
                return np.linalg.solve(matrix, right_hand_side)
            except np.linalg.LinAlgError:
                matrix = matrix + jitter * np.eye(matrix.shape[0])
                jitter *= 10.0
        # This path is diagnostic insurance for exceptionally ill-conditioned
        # streams; it remains bounded and does not retain any extra state.
        return np.linalg.lstsq(matrix, right_hand_side, rcond=None)[0]

    def update(self, features: FloatArray, target: FloatArray, prediction: FloatArray) -> None:
        values = vector("features", features, self.input_size)
        outcome = vector("target", target, self.output_size)
        vector("prediction", prediction, self.output_size)
        probe_projection = self.probes.T @ values
        self.feature_diagonal += np.square(values)
        self.range_statistic += np.outer(values, probe_projection)
        self.probe_covariance += np.outer(probe_projection, probe_projection)
        self.cross_target += np.outer(values, outcome)
        self.sample_count[0] += 1.0
        self._solve_weights()

    @property
    def persistent_arrays(self) -> tuple[np.ndarray, ...]:
        return (
            self.probes, self.feature_diagonal, self.range_statistic,
            self.probe_covariance, self.cross_target, self.weights,
            self.sample_count,
        )

    @property
    def state_nbytes(self) -> int:
        return sum(array.nbytes for array in self.persistent_arrays)

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "algorithm": "nystrom_covariance_memory",
            "rank": self.rank,
            "samples_in_cumulative_statistics": int(self.sample_count[0]),
            "stored_raw_examples": 0,
            "recruitment": False,
            "forgetting_factor": 1.0,
            "gradients": False,
        }

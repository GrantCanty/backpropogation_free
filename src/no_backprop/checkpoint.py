"""Explicit NumPy checkpoints for bounded-memory online learners."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from no_backprop.eligibility import EligibilityReservoir
from no_backprop.protocol import ProtocolError
from no_backprop.readouts import (
    BlockRLSReadout,
    CumulativeMemoryReadout,
    CumulativeMaturityReadout,
    DiagonalRLSReadout,
    FastSlowLMSReadout,
    KeyValueMaturityReadout,
    ProtectedFastSlowReadout,
    PrototypeReadout,
    RLSReadout,
)
from no_backprop.reservoir import OnlineReservoir


def _copy_array(target: np.ndarray, source: np.ndarray, name: str) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"checkpoint {name} has shape {source.shape}; expected {target.shape}"
        )
    np.copyto(target, source)


def save_checkpoint(learner: OnlineReservoir, destination: str | Path) -> Path:
    """Atomically save all persistent model and plasticity state."""

    if learner._pending_prediction is not None:
        raise ProtocolError("cannot checkpoint between predict and learn")
    arrays: dict[str, np.ndarray] = {
        "input_weights": learner.input_weights,
        "recurrent_weights": learner.recurrent_weights,
        "bias": learner.bias,
        "state": learner.state,
    }
    if isinstance(learner.readout, CumulativeMaturityReadout):
        maturity_names = (
            "expanded_weights",
            "inverse_correlation",
            "neuron_centers",
            "neuron_active",
            "neuron_evidence",
            "neuron_labels",
            "neuron_recruitment_entropy",
            "active_count",
            "sample_count",
            "correct_entropy_sum",
            "correct_entropy_count",
            "error_count",
            "recruitment_candidate_count",
            "entropy_rejection_count",
            "proximity_rejection_count",
        )
        arrays.update(
            {
                f"readout_maturity_{name}": getattr(learner.readout, name)
                for name in maturity_names
            }
        )
        if isinstance(learner.readout, KeyValueMaturityReadout):
            for name in ("key_weight", "key_m2", "key_variance"):
                arrays[f"readout_maturity_{name}"] = getattr(
                    learner.readout, name
                )
    elif isinstance(learner.readout, CumulativeMemoryReadout):
        arrays.update(
            {
                "readout_slow_weights": learner.readout.slow_weights,
                "readout_slow_inverse_correlation": (
                    learner.readout.slow_inverse_correlation
                ),
                "readout_semantic_centroids": learner.readout.semantic_centroids,
                "readout_semantic_counts": learner.readout.semantic_counts,
                "readout_exception_centroids": learner.readout.exception_centroids,
                "readout_exception_counts": learner.readout.exception_counts,
                "readout_rank_trials": learner.readout.rank_trials,
                "readout_rank_correct": learner.readout.rank_correct,
                "readout_sample_count": learner.readout.sample_count,
                "readout_rank_update_count": learner.readout.rank_update_count,
                "readout_selection_counts": learner.readout.selection_counts,
            }
        )
    elif isinstance(learner.readout, FastSlowLMSReadout):
        arrays["readout_slow_weights"] = learner.readout.slow_weights
        arrays["readout_fast_weights"] = learner.readout.fast_weights
    elif isinstance(learner.readout, ProtectedFastSlowReadout):
        arrays["readout_slow_centroids"] = learner.readout.slow_memory.centroids
        arrays["readout_slow_counts"] = learner.readout.slow_memory.counts
        arrays["readout_fast_weights"] = learner.readout.fast_weights
    elif isinstance(learner.readout, PrototypeReadout):
        arrays["readout_centroids"] = learner.readout.centroids
        arrays["readout_counts"] = learner.readout.counts
    else:
        arrays["readout_weights"] = learner.readout.weights
    if isinstance(learner.readout, RLSReadout):
        arrays["readout_inverse_correlation"] = learner.readout.inverse_correlation
    elif isinstance(learner.readout, DiagonalRLSReadout):
        arrays["readout_inverse_diagonal"] = learner.readout.inverse_diagonal
    elif isinstance(learner.readout, BlockRLSReadout):
        arrays.update(
            {
                f"readout_inverse_block_{index}": block
                for index, block in enumerate(learner.readout.inverse_blocks)
            }
        )
    if isinstance(learner, EligibilityReservoir):
        arrays.update(
            {
                "recurrent_eligibility": learner.recurrent_eligibility,
                "input_eligibility": learner.input_eligibility,
                "feedback_weights": learner.feedback_weights,
            }
        )

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return path


def restore_checkpoint(learner: OnlineReservoir, source: str | Path) -> None:
    """Restore a checkpoint into a learner with the same architecture."""

    if learner._pending_prediction is not None:
        raise ProtocolError("cannot restore between predict and learn")
    with np.load(Path(source), allow_pickle=False) as arrays:
        _copy_array(learner.input_weights, arrays["input_weights"], "input_weights")
        _copy_array(
            learner.recurrent_weights,
            arrays["recurrent_weights"],
            "recurrent_weights",
        )
        _copy_array(learner.bias, arrays["bias"], "bias")
        _copy_array(learner.state, arrays["state"], "state")
        if isinstance(learner.readout, CumulativeMaturityReadout):
            maturity_names = (
                "expanded_weights",
                "inverse_correlation",
                "neuron_centers",
                "neuron_active",
                "neuron_evidence",
                "neuron_labels",
                "neuron_recruitment_entropy",
                "active_count",
                "sample_count",
                "correct_entropy_sum",
                "correct_entropy_count",
                "error_count",
                "recruitment_candidate_count",
                "entropy_rejection_count",
                "proximity_rejection_count",
            )
            for name in maturity_names:
                checkpoint_name = f"readout_maturity_{name}"
                _copy_array(
                    getattr(learner.readout, name),
                    arrays[checkpoint_name],
                    checkpoint_name,
                )
            if isinstance(learner.readout, KeyValueMaturityReadout):
                for name in ("key_weight", "key_m2", "key_variance"):
                    checkpoint_name = f"readout_maturity_{name}"
                    _copy_array(
                        getattr(learner.readout, name),
                        arrays[checkpoint_name],
                        checkpoint_name,
                    )
        elif isinstance(learner.readout, CumulativeMemoryReadout):
            cumulative_arrays = (
                ("slow_weights", "readout_slow_weights"),
                (
                    "slow_inverse_correlation",
                    "readout_slow_inverse_correlation",
                ),
                ("semantic_centroids", "readout_semantic_centroids"),
                ("semantic_counts", "readout_semantic_counts"),
                ("exception_centroids", "readout_exception_centroids"),
                ("exception_counts", "readout_exception_counts"),
                ("rank_trials", "readout_rank_trials"),
                ("rank_correct", "readout_rank_correct"),
                ("sample_count", "readout_sample_count"),
                ("rank_update_count", "readout_rank_update_count"),
                ("selection_counts", "readout_selection_counts"),
            )
            for attribute, name in cumulative_arrays:
                _copy_array(getattr(learner.readout, attribute), arrays[name], name)
        elif isinstance(learner.readout, FastSlowLMSReadout):
            _copy_array(
                learner.readout.slow_weights,
                arrays["readout_slow_weights"],
                "readout_slow_weights",
            )
            _copy_array(
                learner.readout.fast_weights,
                arrays["readout_fast_weights"],
                "readout_fast_weights",
            )
        elif isinstance(learner.readout, ProtectedFastSlowReadout):
            _copy_array(
                learner.readout.slow_memory.centroids,
                arrays["readout_slow_centroids"],
                "readout_slow_centroids",
            )
            _copy_array(
                learner.readout.slow_memory.counts,
                arrays["readout_slow_counts"],
                "readout_slow_counts",
            )
            _copy_array(
                learner.readout.fast_weights,
                arrays["readout_fast_weights"],
                "readout_fast_weights",
            )
        elif isinstance(learner.readout, PrototypeReadout):
            _copy_array(
                learner.readout.centroids,
                arrays["readout_centroids"],
                "readout_centroids",
            )
            _copy_array(
                learner.readout.counts,
                arrays["readout_counts"],
                "readout_counts",
            )
        else:
            _copy_array(
                learner.readout.weights, arrays["readout_weights"], "readout_weights"
            )
        if isinstance(learner.readout, RLSReadout):
            _copy_array(
                learner.readout.inverse_correlation,
                arrays["readout_inverse_correlation"],
                "readout_inverse_correlation",
            )
        elif isinstance(learner.readout, DiagonalRLSReadout):
            _copy_array(
                learner.readout.inverse_diagonal,
                arrays["readout_inverse_diagonal"],
                "readout_inverse_diagonal",
            )
        elif isinstance(learner.readout, BlockRLSReadout):
            for index, block in enumerate(learner.readout.inverse_blocks):
                name = f"readout_inverse_block_{index}"
                _copy_array(block, arrays[name], name)
        if isinstance(learner, EligibilityReservoir):
            _copy_array(
                learner.recurrent_eligibility,
                arrays["recurrent_eligibility"],
                "recurrent_eligibility",
            )
            _copy_array(
                learner.input_eligibility,
                arrays["input_eligibility"],
                "input_eligibility",
            )
            _copy_array(
                learner.feedback_weights,
                arrays["feedback_weights"],
                "feedback_weights",
            )

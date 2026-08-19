"""Explicit NumPy checkpoints for bounded-memory online learners."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from no_backprop.eligibility import EligibilityReservoir
from no_backprop.protocol import ProtocolError
from no_backprop.readouts import FastSlowLMSReadout, RLSReadout
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
    if isinstance(learner.readout, FastSlowLMSReadout):
        arrays["readout_slow_weights"] = learner.readout.slow_weights
        arrays["readout_fast_weights"] = learner.readout.fast_weights
    else:
        arrays["readout_weights"] = learner.readout.weights
    if isinstance(learner.readout, RLSReadout):
        arrays["readout_inverse_correlation"] = learner.readout.inverse_correlation
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
        if isinstance(learner.readout, FastSlowLMSReadout):
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

"""Reproducible experiment runners and result serialization."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from no_backprop.metrics import PrequentialMetrics
from no_backprop.readouts import FrozenReadout, LMSReadout, RLSReadout
from no_backprop.reservoir import OnlineReservoir, ReservoirConfig
from no_backprop.streams import iter_nonstationary_signal


ReadoutKind = Literal["frozen", "lms", "rls"]


@dataclass(frozen=True)
class SignalExperimentConfig:
    steps: int = 3_000
    regime_length: int = 750
    hidden_size: int = 64
    seed: int = 7
    window: int = 100
    lms_learning_rate: float = 0.18
    rls_regularization: float = 1.0
    rls_forgetting_factor: float = 0.998


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def build_signal_learner(
    kind: ReadoutKind, config: SignalExperimentConfig
) -> OnlineReservoir:
    reservoir_config = ReservoirConfig(
        input_size=1,
        hidden_size=config.hidden_size,
        output_size=1,
        seed=config.seed,
    )
    feature_size = config.hidden_size + 1
    if kind == "frozen":
        readout = FrozenReadout(feature_size, 1, seed=config.seed)
    elif kind == "lms":
        readout = LMSReadout(
            feature_size,
            1,
            seed=config.seed,
            learning_rate=config.lms_learning_rate,
        )
    elif kind == "rls":
        readout = RLSReadout(
            feature_size,
            1,
            seed=config.seed,
            regularization=config.rls_regularization,
            forgetting_factor=config.rls_forgetting_factor,
        )
    else:
        raise ValueError(f"unknown readout kind: {kind}")
    return OnlineReservoir(reservoir_config, readout)


def run_signal_model(
    kind: ReadoutKind, config: SignalExperimentConfig
) -> dict[str, Any]:
    learner = build_signal_learner(kind, config)
    metrics = PrequentialMetrics()
    initial_state_bytes = learner.state_nbytes
    rss_before = _process_rss_bytes()
    started = time.perf_counter()
    for event in iter_nonstationary_signal(
        config.steps,
        regime_length=config.regime_length,
        seed=config.seed,
    ):
        prediction = learner.predict(event.observation)
        metrics.record(
            prediction,
            event.target,
            regime=event.regime,
            change_point=event.change_point,
        )
        learner.learn(event.target)
    elapsed = time.perf_counter() - started
    rss_after = _process_rss_bytes()
    summary = metrics.summary(config.window)
    summary.update(
        {
            "model": kind,
            "elapsed_seconds": elapsed,
            "steps_per_second": config.steps / elapsed,
            "state_bytes_before": initial_state_bytes,
            "state_bytes_after": learner.state_nbytes,
            "bounded_state": initial_state_bytes == learner.state_nbytes,
            "rss_delta_bytes": (
                None if rss_before is None or rss_after is None else rss_after - rss_before
            ),
            "segments": metrics.segment_summaries(config.window),
        }
    )
    return summary


def run_signal_experiment(config: SignalExperimentConfig) -> dict[str, Any]:
    models = {
        kind: run_signal_model(kind, config) for kind in ("frozen", "lms", "rls")
    }
    return {
        "experiment": "nonstationary_signal",
        "config": asdict(config),
        "models": models,
    }


def write_json_result(result: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path

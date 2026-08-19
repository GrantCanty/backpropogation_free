"""Multi-seed replication for the local-learning MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from no_backprop.experiment import (
    ContinualExperimentConfig,
    DelayedExperimentConfig,
    run_continual_experiment,
    run_delayed_experiment,
)


@dataclass(frozen=True)
class ReplicationConfig:
    seeds: tuple[int, ...] = (3, 7, 11, 17, 23)
    delayed: DelayedExperimentConfig = DelayedExperimentConfig(
        episodes=1_200, delay=8, hidden_size=48
    )
    continual: ContinualExperimentConfig = ContinualExperimentConfig(
        steps=3_200, context_length=800, hidden_size=48, window=120
    )


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": values,
    }


def _aggregate_models(
    runs: list[dict[str, Any]],
    models: tuple[str, ...],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    return {
        model: {
            metric: _summary([float(run["models"][model][metric]) for run in runs])
            for metric in metrics
        }
        for model in models
    }


def run_replication(config: ReplicationConfig = ReplicationConfig()) -> dict[str, Any]:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    delayed_runs = [
        run_delayed_experiment(replace(config.delayed, seed=seed))
        for seed in config.seeds
    ]
    continual_runs = [
        run_continual_experiment(replace(config.continual, seed=seed))
        for seed in config.seeds
    ]
    delayed = _aggregate_models(
        delayed_runs,
        ("fixed", "eligibility"),
        ("mse", "tail_mse", "accuracy", "tail_accuracy"),
    )
    continual = _aggregate_models(
        continual_runs,
        ("fixed", "eligibility", "gated", "fast_slow"),
        ("accuracy", "tail_accuracy", "retention_delta", "mse"),
    )
    delayed["eligibility_mse_improvement"] = _summary(
        [
            float(run["models"]["fixed"]["mse"])
            - float(run["models"]["eligibility"]["mse"])
            for run in delayed_runs
        ]
    )
    continual["fast_slow_retention_improvement"] = _summary(
        [
            float(run["models"]["fast_slow"]["retention_delta"])
            - float(run["models"]["fixed"]["retention_delta"])
            for run in continual_runs
        ]
    )
    continual["fast_slow_accuracy_tradeoff"] = _summary(
        [
            float(run["models"]["fast_slow"]["accuracy"])
            - float(run["models"]["fixed"]["accuracy"])
            for run in continual_runs
        ]
    )
    return {
        "experiment": "mvp_replication",
        "config": {
            "seeds": list(config.seeds),
            "delayed": asdict(config.delayed),
            "continual": asdict(config.continual),
        },
        "delayed": delayed,
        "continual": continual,
    }

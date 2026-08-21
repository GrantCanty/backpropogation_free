"""Reusable factory-driven online classification experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from continual_core.evaluation import (
    TargetEncoder,
    evaluate_classification_locked,
    train_prequential,
)
from continual_core.protocols import FloatArray, TaskAdapter
from continual_core.results import result_envelope


@dataclass(frozen=True)
class MethodSetup:
    """Everything the neutral runner needs to exercise one learner."""

    factory: Callable[[], object]
    adapter: TaskAdapter
    assumptions: Mapping[str, Any] = field(default_factory=dict)


def run_method(
    *,
    experiment: str,
    method: str,
    setup: MethodSetup,
    training_events: Iterable[tuple[FloatArray, FloatArray]],
    evaluation_observations: Iterable[FloatArray],
    evaluation_labels: Iterable[Any],
    encode_target: TargetEncoder,
    seed: int,
) -> dict[str, Any]:
    """Train and lock-evaluate one injected method on a matched stream."""

    learner = setup.factory()
    training = train_prequential(learner, training_events, setup.adapter)
    evaluation = evaluate_classification_locked(
        learner,
        evaluation_observations,
        evaluation_labels,
        setup.adapter,
        encode_target,
    )
    diagnostics = dict(getattr(learner, "diagnostics", {}))
    return result_envelope(
        experiment=experiment,
        method=method,
        seed=seed,
        assumptions=setup.assumptions,
        payload={
            "training": training,
            "evaluation": evaluation,
            "diagnostics": diagnostics,
        },
    )


def run_comparison(
    *,
    experiment: str,
    methods: Mapping[str, MethodSetup],
    training_events: Sequence[tuple[FloatArray, FloatArray]],
    evaluation_observations: Sequence[FloatArray],
    evaluation_labels: Sequence[Any],
    encode_target: TargetEncoder,
    seed: int,
) -> dict[str, Any]:
    """Run all methods over the same materialized events and evaluation set."""

    return {
        "schema_version": 1,
        "experiment": experiment,
        "seed": seed,
        "methods": {
            name: run_method(
                experiment=experiment,
                method=name,
                setup=setup,
                training_events=training_events,
                evaluation_observations=evaluation_observations,
                evaluation_labels=evaluation_labels,
                encode_target=encode_target,
                seed=seed,
            )
            for name, setup in methods.items()
        },
    }

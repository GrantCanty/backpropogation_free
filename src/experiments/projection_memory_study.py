"""Method-neutral 2x2 study for frozen projections and covariance memories.

The runner accepts already prepared raw events.  Dataset loading and stream
construction therefore remain outside this proposal-specific experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from baselines.os_elm import OSELMFeatureMap
from baselines.rls import RLSReadout
from continual_core.datasets.classification import (
    augment_image_split,
    load_classification_split,
)
from continual_core.datasets.digits import build_digits_segments
from continual_core.evaluation import FeatureMapAdapter, train_classification_profiled
from continual_core.metrics import (
    classification_plasticity_summary,
    sample_efficiency_steps,
)
from continual_core.protocols import FloatArray
from continual_core.results import artifact_matches, write_json_result
from methods.nystrom_memory import NystromCovarianceReadout
from methods.structured_projection import SparseSignedFeatureMap


MEASUREMENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ProjectionMemoryConfig:
    """Declarative knobs shared by development and confirmatory campaigns."""

    input_size: int
    hidden_size: int = 256
    regularization: float = 1.0
    fan_ins: tuple[int, ...] = (4, 8, 16, 32)
    ranks: tuple[int, ...] = (8, 16, 32, 64)
    nyström_regularizations: tuple[float, ...] = (0.1, 1.0, 10.0)
    development_seeds: tuple[int, ...] = (11, 23, 37, 41, 53)
    confirmatory_seeds: tuple[int, ...] = tuple(range(100, 120))
    max_sparse_accuracy_loss: float = 0.01
    max_nystrom_accuracy_loss: float = 0.01

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if set(self.development_seeds) & set(self.confirmatory_seeds):
            raise ValueError("development and confirmatory seeds must be disjoint")
        if any(fan <= 0 or fan > self.input_size for fan in self.fan_ins):
            raise ValueError("fan-ins must be in [1, input_size]")
        if any(rank <= 0 for rank in self.ranks):
            raise ValueError("ranks must be positive")


def materialize_projection_problem(
    *,
    dataset: str,
    seed: int,
    protocol: str,
    events_per_segment: int,
    test_per_class: int,
    dataset_path: str | Path | None = None,
    allow_download: bool = False,
    dataset_cache: str | Path | None = None,
    augmentation_copies: int = 1,
    augmentation_max_shift: int = 1,
    augmentation_noise_std: float = 0.03,
    recurring_visits: int = 2,
) -> tuple[
    list[list[tuple[FloatArray, FloatArray]]],
    dict[str, tuple[list[FloatArray], list[Any]]],
    int,
]:
    """Build one public, method-neutral raw classification problem."""

    split = load_classification_split(
        dataset,
        test_per_class=test_per_class,
        seed=seed,
        dataset_path=dataset_path,
        allow_download=allow_download,
        cache_directory=dataset_cache,
    )
    split = augment_image_split(
        split,
        copies=augmentation_copies,
        max_shift=augmentation_max_shift,
        noise_std=augmentation_noise_std,
        seed=seed + 10_000,
    )
    stream = build_digits_segments(
        split.train_labels,
        protocol=protocol,  # type: ignore[arg-type]
        passes=(recurring_visits if protocol == "class_recurring" else 1),
        seed=seed + 20_000,
    )
    selected = [segment.indices[:events_per_segment] for segment in stream]
    observations = split.train_images.reshape(len(split.train_images), -1)
    labels = split.train_labels.astype(int)
    classes = len(np.unique(labels))
    identity = np.eye(classes, dtype=np.float64)
    segments = [
        [
            (observations[int(index)], identity[labels[int(index)]])
            for index in indices
        ]
        for indices in selected
    ]
    test_observations = split.test_images.reshape(len(split.test_images), -1)
    evaluation: dict[str, tuple[list[FloatArray], list[Any]]] = {
        "all": (
            list(test_observations),
            split.test_labels.astype(int).tolist(),
        )
    }
    for label in np.unique(split.test_labels):
        indices = np.flatnonzero(split.test_labels == label)
        evaluation[f"class_{int(label)}"] = (
            [test_observations[int(index)] for index in indices],
            [int(label)] * len(indices),
        )
    return segments, evaluation, int(observations.shape[1])


def configuration_hash(config: ProjectionMemoryConfig, *, condition: str,
                       parameters: Mapping[str, Any] | None = None,
                       problem_hash: str | None = None,
                       protocol: str | None = None) -> str:
    payload = json.dumps({"config": asdict(config), "condition": condition,
                          "parameters": dict(parameters or {}),
                          "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
                          "problem_hash": problem_hash,
                          "protocol": protocol},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def projection_problem_fingerprint(
    segments: Sequence[Sequence[tuple[FloatArray, FloatArray]]],
    evaluation_sets: Mapping[str, tuple[list[FloatArray], list[Any]]],
) -> str:
    """Hash exact ordered raw events and locked evaluation observations."""

    digest = hashlib.sha256()
    for segment in segments:
        digest.update(len(segment).to_bytes(8, "little"))
        for observation, target in segment:
            digest.update(np.asarray(observation, dtype="<f8").tobytes())
            digest.update(np.asarray(target, dtype="<f8").tobytes())
    for name in sorted(evaluation_sets):
        observations, labels = evaluation_sets[name]
        digest.update(name.encode("utf-8"))
        for observation, label in zip(observations, labels):
            digest.update(np.asarray(observation, dtype="<f8").tobytes())
            digest.update(str(label).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def runtime_environment(timing_session_id: str | None = None) -> dict[str, Any]:
    """Record implementation and host details needed to interpret timing."""

    thread_variables = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    return {
        "timing_session_id": timing_session_id or platform.node(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name) for name in thread_variables
        },
        "dtype": "float64",
        "backend": "numpy",
        "batching": "event-wise",
    }


def feature_map_factory(kind: str, config: ProjectionMemoryConfig, *, seed: int,
                        fan_in: int | None = None) -> object:
    if kind == "dense":
        return OSELMFeatureMap(config.input_size, config.hidden_size, seed=seed)
    if kind == "sparse":
        if fan_in is None:
            raise ValueError("sparse feature maps require fan_in")
        return SparseSignedFeatureMap(config.input_size, config.hidden_size, fan_in, seed=seed)
    raise ValueError(f"unknown feature map {kind!r}")


def readout_factory(kind: str, width: int, output_size: int, config: ProjectionMemoryConfig,
                    *, seed: int, rank: int | None = None,
                    regularization: float | None = None) -> object:
    ridge = config.regularization if regularization is None else regularization
    if kind == "exact":
        return RLSReadout(width, output_size, seed=seed, regularization=ridge,
                          forgetting_factor=1.0)
    if kind == "nystrom":
        if rank is None:
            raise ValueError("Nyström readouts require rank")
        return NystromCovarianceReadout(width, output_size, rank=rank, seed=seed,
                                        regularization=ridge)
    raise ValueError(f"unknown readout {kind!r}")


def run_condition(*, condition: str, feature_kind: str, readout_kind: str,
                  config: ProjectionMemoryConfig, seed: int,
                  segments: list[list[tuple[FloatArray, FloatArray]]],
                  evaluation_sets: Mapping[str, tuple[list[FloatArray], list[Any]]],
                  fan_in: int | None = None, rank: int | None = None,
                  regularization: float | None = None,
                  protocol: str | None = None,
                  problem_hash: str | None = None,
                  environment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    started = perf_counter()
    feature_map = feature_map_factory(feature_kind, config, seed=seed, fan_in=fan_in)
    width = feature_map.output_size  # type: ignore[attr-defined]
    classes = len(evaluation_sets) - 1
    learner = readout_factory(readout_kind, width, classes, config, seed=seed,
                              rank=rank, regularization=regularization)
    adapter = FeatureMapAdapter(feature_map)
    target = lambda label: np.eye(classes, dtype=np.float64)[int(label)]
    training = train_classification_profiled(
        learner, segments, adapter, dict(evaluation_sets), target,
        sample_efficiency_steps=sample_efficiency_steps(
            sum(len(segment) for segment in segments)
        ),
    )
    feature_bytes = int(feature_map.state_nbytes)  # type: ignore[attr-defined]
    solver_bytes = int(learner.state_nbytes)  # type: ignore[attr-defined]
    transform_values = np.asarray(adapter.transform_durations_ns, dtype=np.float64) / 1_000.0
    solver_values = np.asarray(adapter.learner_update_durations_ns, dtype=np.float64) / 1_000.0
    parameters = {"feature": feature_kind, "readout": readout_kind,
                  "fan_in": fan_in, "rank": rank,
                  "regularization": regularization}
    return {
        "schema_version": 1,
        "experiment": "structured_projection_nystrom_memory",
        "condition": condition,
        "protocol": protocol,
        "seed": seed,
        "configuration_hash": configuration_hash(config, condition=condition,
                                                  parameters=parameters,
                                                  problem_hash=problem_hash,
                                                  protocol=protocol),
        "matched_problem_sha256": problem_hash,
        "completed": True,
        "parameters": parameters,
        "resources": {
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "representation_bytes": feature_bytes,
            "solver_bytes": solver_bytes,
            "total_persistent_bytes": feature_bytes + solver_bytes,
            "peak_transient_training_bytes": training["peak_traced_training_bytes"],
            "transform_latency": {
                "mean_microseconds": float(np.mean(transform_values)) if len(transform_values) else 0.0,
                "median_microseconds": float(np.median(transform_values)) if len(transform_values) else 0.0,
                "samples": int(len(transform_values)),
            },
            "solver_update_latency": {
                "mean_microseconds": float(np.mean(solver_values)) if len(solver_values) else 0.0,
                "median_microseconds": float(np.median(solver_values)) if len(solver_values) else 0.0,
                "p95_microseconds": float(np.percentile(solver_values, 95)) if len(solver_values) else 0.0,
            },
            "prediction_latency": training["prediction_latency"],
            "end_to_end_event_latency": training["event_latency"],
            "throughput_events_per_second": training[
                "stream_samples_per_second"
            ],
            "wall_throughput_including_checkpoints": training[
                "samples_per_second"
            ],
        },
        "elapsed_seconds": perf_counter() - started,
        "feature_map_diagnostics": dict(getattr(feature_map, "diagnostics", {})),
        "diagnostics": dict(getattr(learner, "diagnostics", {})),
        "continual_learning_metrics": classification_plasticity_summary(training),
        "runtime_environment": dict(environment or {}),
        "training": training,
    }


def run_projection_memory_study(*, config: ProjectionMemoryConfig,
                                seeds: Sequence[int],
                                conditions: Mapping[str, Mapping[str, Any]],
                                segments_by_seed: Mapping[int, list[list[tuple[FloatArray, FloatArray]]]],
                                evaluation_by_seed: Mapping[int, Mapping[str, tuple[list[FloatArray], list[Any]]]],
                                output: str | Path | None = None,
                                resume: bool = True,
                                progress: Callable[[str], None] | None = None,
                                protocol: str | None = None,
                                timing_session_id: str | None = None,
                                rotate_condition_order: bool = True) -> dict[str, Any]:
    """Run paired conditions and optionally persist one atomic artifact per seed."""
    root = Path(output) if output is not None else None
    runs: list[dict[str, Any]] = []
    environment = runtime_environment(timing_session_id)
    condition_items = list(conditions.items())
    for seed_index, seed in enumerate(seeds):
        problem_hash = projection_problem_fingerprint(
            segments_by_seed[seed], evaluation_by_seed[seed]
        )
        if rotate_condition_order and condition_items:
            shift = seed_index % len(condition_items)
            ordered_conditions = condition_items[shift:] + condition_items[:shift]
        else:
            ordered_conditions = condition_items
        for name, parameters in ordered_conditions:
            path = root / "raw" / f"{name}__seed_{seed}.json" if root else None
            expected_hash = configuration_hash(config, condition=name,
                                               parameters=parameters,
                                               problem_hash=problem_hash,
                                               protocol=protocol)
            if resume and path and artifact_matches(path, experiment="structured_projection_nystrom_memory",
                                                    seed=seed, configuration_hash=expected_hash):
                runs.append(json.loads(path.read_text(encoding="utf-8")))
                if progress:
                    progress(f"resumed condition={name} seed={seed}")
                continue
            run = run_condition(condition=name, config=config, seed=seed,
                                segments=segments_by_seed[seed],
                                evaluation_sets=evaluation_by_seed[seed],
                                protocol=protocol, problem_hash=problem_hash,
                                environment=environment, **parameters)
            if path:
                write_json_result(run, path)
            runs.append(run)
            if progress:
                progress(f"completed condition={name} seed={seed}")
    result = {"schema_version": 1, "experiment": "structured_projection_nystrom_memory",
              "configuration": asdict(config), "runs": runs,
              "conditions": dict(conditions),
              "protocol": protocol,
              "runtime_environment": environment,
              "condition_order_rotated_by_seed": rotate_condition_order,
              "resume_artifacts": str(root / "raw") if root else None}
    if root:
        write_json_result(result, root / "comparison.json")
        write_projection_report(result, root / "REPORT.md")
    return result


def write_projection_report(result: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a compact reproducible report while retaining raw per-seed JSON."""
    lines = ["# Structured Projection and Nyström Memory Study", "",
             f"Runs: {len(result.get('runs', []))}", ""]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in result.get("runs", []):
        grouped.setdefault(str(run["condition"]), []).append(run)
    for condition, runs in sorted(grouped.items()):
        accuracy = np.mean([run["training"]["online_accuracy"] for run in runs])
        memory = np.mean([run["resources"]["total_persistent_bytes"] for run in runs])
        lines.append(f"## {condition}")
        lines.append("")
        lines.append(f"- mean online accuracy: {accuracy:.6f}")
        lines.append(f"- mean persistent bytes: {memory:.0f}")
        lines.append(f"- seeds: {', '.join(str(run['seed']) for run in runs)}")
        lines.append("")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def select_sparse_gate(development_runs: Sequence[Mapping[str, Any]], *, dense_name: str = "dense_exact") -> dict[str, Any]:
    """Apply the predeclared memory, quality, and latency gates to WP1 runs."""
    dense = [run for run in development_runs if run["condition"] == dense_name]
    if not dense:
        raise ValueError(f"missing dense reference {dense_name!r}")
    dense_accuracy = float(np.mean([run["training"]["online_accuracy"] for run in dense]))
    dense_memory = float(np.mean([run["resources"]["total_persistent_bytes"] for run in dense]))
    dense_latency = float(np.mean([run["resources"]["end_to_end_event_latency"]["median_microseconds"] for run in dense]))
    candidates = []
    for run in development_runs:
        if run["condition"] == dense_name or run["parameters"].get("readout") != "exact":
            continue
        accuracy = float(run["training"]["online_accuracy"])
        memory = float(run["resources"]["total_persistent_bytes"])
        latency = float(run["resources"]["end_to_end_event_latency"]["median_microseconds"])
        if memory <= 0.75 * dense_memory and accuracy >= dense_accuracy - 0.01 and latency <= dense_latency:
            candidates.append(run)
    winner = min(candidates, key=lambda item: (item["parameters"].get("fan_in", 10**9),
                                                 item["resources"]["end_to_end_event_latency"]["median_microseconds"]), default=None)
    return {"passed": winner is not None, "selected": winner["parameters"] if winner else None,
            "candidate_count": len(candidates), "gate": {"memory_fraction": 0.75,
            "accuracy_loss": 0.01, "latency": "median <= dense"}}

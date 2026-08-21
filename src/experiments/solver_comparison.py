"""Controlled OS-ELM output-solver comparison on local 8x8 digits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from baselines.lms import LMSReadout
from baselines.os_elm import OSELMFeatureMap
from baselines.rls import BlockRLSReadout, DiagonalRLSReadout, RLSReadout
from baselines.rpls import RecursivePLSReadout
from continual_core.datasets.classification import (
    augment_image_split,
    load_classification_split,
)
from continual_core.datasets.digits import build_digits_segments
from continual_core.evaluation import (
    DirectUpdateAdapter,
    FeatureSubsetAdapter,
    train_classification_profiled,
)
from continual_core.protocols import FloatArray, TaskAdapter
from continual_core.results import write_json_result
from methods.covariance_sketch import FrequentDirectionsRidgeReadout


@dataclass(frozen=True)
class SolverComparisonConfig:
    dataset: str = "digits"
    dataset_path: str | None = None
    allow_download: bool = False
    dataset_cache_directory: str | None = None
    hidden_size: int = 64
    test_per_class: int = 40
    augmentation_copies: int = 1
    augmentation_max_shift: int = 1
    augmentation_noise_std: float = 0.03
    development_seeds: tuple[int, ...] = (11, 23)
    heldout_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    development_events_per_segment: int = 25
    heldout_events_per_segment: int | None = None
    regularization_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    lms_learning_rates: tuple[float, ...] = (0.03, 0.1, 0.3)
    block_sizes: tuple[int, ...] = (4, 8, 16)
    rpls_components: tuple[int, ...] = (4, 8, 16)
    sketch_ranks: tuple[int, ...] = (4, 8, 16)

    def __post_init__(self) -> None:
        if self.hidden_size < 4:
            raise ValueError("hidden_size must be at least four")
        if set(self.development_seeds) & set(self.heldout_seeds):
            raise ValueError("development and held-out seeds must be disjoint")
        if not self.development_seeds or not self.heldout_seeds:
            raise ValueError("development and held-out seeds cannot be empty")
        if self.development_events_per_segment <= 0:
            raise ValueError("development_events_per_segment must be positive")
        if (
            self.heldout_events_per_segment is not None
            and self.heldout_events_per_segment <= 0
        ):
            raise ValueError("heldout_events_per_segment must be positive")


@dataclass(frozen=True)
class _SolverSetup:
    factory: Callable[[], object]
    adapter: TaskAdapter
    parameters: Mapping[str, Any]
    representation_coordinates: int


def _target(label: int, classes: int = 10) -> FloatArray:
    encoded = np.zeros(classes, dtype=np.float64)
    encoded[int(label)] = 1.0
    return encoded


def _transform_images(
    feature_map: OSELMFeatureMap, images: np.ndarray
) -> list[FloatArray]:
    return [feature_map.transform(image.reshape(-1)) for image in images]


def _materialize_problem(
    config: SolverComparisonConfig,
    *,
    seed: int,
    protocol: str,
    development: bool,
) -> tuple[
    list[list[tuple[FloatArray, FloatArray]]],
    dict[str, tuple[list[FloatArray], list[int]]],
    OSELMFeatureMap,
]:
    split = load_classification_split(
        config.dataset,
        test_per_class=config.test_per_class,
        seed=seed,
        dataset_path=config.dataset_path,
        allow_download=config.allow_download,
        cache_directory=config.dataset_cache_directory,
    )
    split = augment_image_split(
        split,
        copies=config.augmentation_copies,
        max_shift=config.augmentation_max_shift,
        noise_std=config.augmentation_noise_std,
        seed=seed + 10_000,
    )
    input_size = int(np.prod(split.train_images.shape[1:]))
    classes = int(len(np.unique(split.train_labels)))
    feature_map = OSELMFeatureMap(input_size, config.hidden_size, seed=seed)
    test_features = _transform_images(feature_map, split.test_images)
    segments = build_digits_segments(
        split.train_labels,
        protocol=protocol,  # type: ignore[arg-type]
        seed=seed + 20_000,
    )
    event_segments: list[list[tuple[FloatArray, FloatArray]]] = []
    selected_segments: list[np.ndarray] = []
    for segment in segments:
        indices = segment.indices
        limit = (
            config.development_events_per_segment
            if development
            else config.heldout_events_per_segment
        )
        if limit is not None:
            indices = indices[:limit]
        selected_segments.append(indices)
    selected_indices = np.unique(np.concatenate(selected_segments))
    training_features = {
        int(index): feature_map.transform(split.train_images[index].reshape(-1))
        for index in selected_indices
    }
    for indices in selected_segments:
        event_segments.append(
            [
                (
                    training_features[int(index)],
                    _target(int(split.train_labels[index]), classes),
                )
                for index in indices
            ]
        )
    evaluation_sets: dict[str, tuple[list[FloatArray], list[int]]] = {
        "all": (test_features, split.test_labels.astype(int).tolist())
    }
    for label in range(classes):
        indices = np.flatnonzero(split.test_labels == label)
        evaluation_sets[f"class_{label}"] = (
            [test_features[int(index)] for index in indices],
            [label] * len(indices),
        )
    return event_segments, evaluation_sets, feature_map


def _base_setup(
    name: str,
    parameters: Mapping[str, Any],
    *,
    width: int,
    classes: int = 10,
) -> _SolverSetup:
    values = dict(parameters)
    seed = int(values.pop("seed", 0))
    if name == "lms":
        factory = lambda: LMSReadout(
            width,
            classes,
            seed=seed,
            learning_rate=float(values["learning_rate"]),
            normalized=True,
        )
    elif name == "exact_rls":
        factory = lambda: RLSReadout(
            width,
            classes,
            seed=seed,
            regularization=float(values["regularization"]),
            forgetting_factor=1.0,
        )
    elif name == "diagonal_rls":
        factory = lambda: DiagonalRLSReadout(
            width,
            classes,
            seed=seed,
            regularization=float(values["regularization"]),
            forgetting_factor=1.0,
        )
    elif name == "block_rls":
        factory = lambda: BlockRLSReadout(
            width,
            classes,
            seed=seed,
            regularization=float(values["regularization"]),
            forgetting_factor=1.0,
            block_size=int(values["block_size"]),
        )
    elif name == "rpls":
        factory = lambda: RecursivePLSReadout(
            width,
            classes,
            seed=seed,
            regularization=float(values["regularization"]),
            components=int(values["components"]),
        )
    elif name == "fd_ridge":
        factory = lambda: FrequentDirectionsRidgeReadout(
            width,
            classes,
            seed=seed,
            regularization=float(values["regularization"]),
            sketch_rank=int(values["sketch_rank"]),
        )
    else:
        raise ValueError(f"unknown solver {name!r}")
    return _SolverSetup(
        factory=factory,
        adapter=DirectUpdateAdapter(),
        parameters=dict(parameters),
        representation_coordinates=width,
    )


def _candidate_grid(
    config: SolverComparisonConfig,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "lms": [
            {"learning_rate": rate} for rate in config.lms_learning_rates
        ],
        "exact_rls": [
            {"regularization": value}
            for value in config.regularization_grid
        ],
        "diagonal_rls": [
            {"regularization": value}
            for value in config.regularization_grid
        ],
        "block_rls": [
            {"regularization": value, "block_size": size}
            for value in config.regularization_grid
            for size in config.block_sizes
        ],
        "rpls": [
            {"regularization": value, "components": count}
            for value in config.regularization_grid
            for count in config.rpls_components
        ],
        "fd_ridge": [
            {"regularization": value, "sketch_rank": rank}
            for value in config.regularization_grid
            for rank in config.sketch_ranks
            if rank < config.hidden_size + 1
        ],
    }


def _sample_steps(segments: list[list[Any]]) -> tuple[int, ...]:
    total = sum(len(segment) for segment in segments)
    return tuple(
        sorted(
            {
                max(1, int(total * fraction))
                for fraction in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
            }
        )
    )


def _representation_bytes(coordinates: int, input_size: int) -> int:
    hidden = coordinates - 1
    return int(8 * (hidden * input_size + hidden))


def _problem_fingerprint(
    segments: list[list[tuple[FloatArray, FloatArray]]],
    evaluation_sets: Mapping[str, tuple[list[FloatArray], list[int]]],
) -> str:
    """Hash the exact ordered features, targets, and locked evaluation sets."""

    digest = hashlib.sha256()
    for segment in segments:
        digest.update(len(segment).to_bytes(8, "little"))
        for observation, target in segment:
            digest.update(np.asarray(observation, dtype="<f8").tobytes())
            digest.update(np.asarray(target, dtype="<f8").tobytes())
    for name in sorted(evaluation_sets):
        observations, labels = evaluation_sets[name]
        digest.update(name.encode("utf-8"))
        digest.update(len(observations).to_bytes(8, "little"))
        for observation, label in zip(observations, labels):
            digest.update(np.asarray(observation, dtype="<f8").tobytes())
            digest.update(int(label).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _derived_metrics(
    training: Mapping[str, Any], *, protocol: str
) -> dict[str, float | int]:
    checkpoints = training["checkpoints"]
    final_accuracy = float(
        checkpoints[-1]["evaluation_sets"]["all"]["accuracy"]
    )
    efficiency_accuracy = np.asarray(
        [
            point["cumulative_online_accuracy"]
            for point in training["sample_efficiency"]
        ],
        dtype=np.float64,
    )
    efficiency_fraction = np.asarray(
        [
            point["samples"] / training["samples"]
            for point in training["sample_efficiency"]
        ],
        dtype=np.float64,
    )
    efficiency_auc = (
        float(
            np.sum(
                np.diff(efficiency_fraction)
                * (efficiency_accuracy[:-1] + efficiency_accuracy[1:])
                * 0.5
            )
        )
        if len(efficiency_fraction) > 1
        else 0.0
    )
    result: dict[str, float | int] = {
        "online_accuracy": float(training["online_accuracy"]),
        "final_locked_accuracy": final_accuracy,
        "sample_efficiency_auc": efficiency_auc,
        "mean_adaptation_delta": float(
            np.mean(
                [segment["adaptation_delta"] for segment in training["segments"]]
            )
        ),
    }
    all_accuracies = [
        checkpoint["evaluation_sets"]["all"]["accuracy"]
        for checkpoint in checkpoints
    ]
    result["final_drop_from_best"] = float(
        max(all_accuracies) - all_accuracies[-1]
    )
    if protocol == "class_ordered":
        class_count = min(
            len(checkpoints[0]["evaluation_sets"]) - 1,
            len(checkpoints),
        )
        matrix = np.asarray(
            [
                [
                    checkpoint["evaluation_sets"][f"class_{label}"]["accuracy"]
                    for label in range(class_count)
                ]
                for checkpoint in checkpoints[:class_count]
            ],
            dtype=np.float64,
        )
        forgetting = [
            float(np.max(matrix[label:, label]) - matrix[-1, label])
            for label in range(class_count - 1)
        ]
        backward_transfer = [
            float(matrix[-1, label] - matrix[label, label])
            for label in range(class_count - 1)
        ]
        result["average_forgetting"] = float(np.mean(forgetting))
        result["backward_transfer"] = float(np.mean(backward_transfer))
    else:
        result["average_forgetting"] = result["final_drop_from_best"]
        result["backward_transfer"] = float(
            all_accuracies[-1] - all_accuracies[0]
        )
    return result


def _run_setup(
    *,
    name: str,
    setup: _SolverSetup,
    segments: list[list[tuple[FloatArray, FloatArray]]],
    evaluation_sets: dict[str, tuple[list[FloatArray], list[int]]],
    feature_map: OSELMFeatureMap,
    seed: int,
    protocol: str,
    phase: str,
    dataset: str,
) -> dict[str, Any]:
    learner = setup.factory()
    classes = len(evaluation_sets) - 1
    encode_target = lambda label: _target(int(label), classes)
    training = train_classification_profiled(
        learner,
        segments,
        setup.adapter,
        evaluation_sets,
        encode_target,
        sample_efficiency_steps=_sample_steps(segments),
    )
    solver_bytes = int(training["state_bytes_after"])
    representation_bytes = _representation_bytes(
        setup.representation_coordinates, feature_map.input_size
    )
    return {
        "schema_version": 1,
        "experiment": "os_elm_solver_comparison",
        "phase": phase,
        "seed": seed,
        "protocol": protocol,
        "matched_problem_sha256": _problem_fingerprint(
            segments, evaluation_sets
        ),
        "method": name,
        "parameters": dict(setup.parameters),
        "assumptions": {
            "gradients": False,
            "backpropagation": False,
            "replay": False,
            "task_identity": False,
            "forgetting_factor": (
                1.0 if "rls" in name and name != "rpls" else None
            ),
            "fixed_os_elm_representation": True,
            "full_feature_map_state_bytes": feature_map.state_nbytes,
            "dataset": dataset,
        },
        "resources": {
            "solver_state_bytes": solver_bytes,
            "representation_state_bytes": representation_bytes,
            "total_persistent_bytes": solver_bytes + representation_bytes,
            "update_mean_microseconds": training["update_latency"][
                "mean_microseconds"
            ],
            "update_p95_microseconds": training["update_latency"][
                "p95_microseconds"
            ],
            "prediction_mean_microseconds": training["prediction_latency"][
                "mean_microseconds"
            ],
            "samples_per_second": training["samples_per_second"],
            "peak_traced_training_bytes": training[
                "peak_traced_training_bytes"
            ],
            "peak_process_rss_bytes": training["peak_process_rss_bytes"],
            "peak_memory_note": training["peak_memory_note"],
        },
        "numerical_stability": {
            "nonfinite_state_values": training["nonfinite_state_values"],
            "maximum_absolute_state_value": training[
                "maximum_absolute_state_value"
            ],
            "bounded_state": training["bounded_state"],
        },
        "metrics": _derived_metrics(training, protocol=protocol),
        "training": training,
        "diagnostics": dict(getattr(learner, "diagnostics", {})),
    }


def _development_score(runs: list[Mapping[str, Any]]) -> float:
    return float(
        np.mean(
            [
                0.5 * run["metrics"]["online_accuracy"]
                + 0.5 * run["metrics"]["final_locked_accuracy"]
                for run in runs
            ]
        )
    )


def _tune(config: SolverComparisonConfig) -> tuple[
    dict[str, dict[str, Any]], list[dict[str, Any]]
]:
    selected: dict[str, dict[str, Any]] = {}
    trials: list[dict[str, Any]] = []
    width = config.hidden_size + 1
    problems = {
        (seed, protocol): _materialize_problem(
            config, seed=seed, protocol=protocol, development=True
        )
        for seed in config.development_seeds
        for protocol in ("shuffled_augmented", "class_ordered")
    }
    for name, configurations in _candidate_grid(config).items():
        best_score = -float("inf")
        best_parameters: dict[str, Any] | None = None
        for parameters in configurations:
            runs: list[dict[str, Any]] = []
            for seed in config.development_seeds:
                for protocol in ("shuffled_augmented", "class_ordered"):
                    segments, evaluation_sets, feature_map = problems[
                        (seed, protocol)
                    ]
                    setup = _base_setup(
                        name,
                        {**parameters, "seed": seed},
                        width=width,
                        classes=len(evaluation_sets) - 1,
                    )
                    runs.append(
                        _run_setup(
                            name=name,
                            setup=setup,
                            segments=segments,
                            evaluation_sets=evaluation_sets,
                            feature_map=feature_map,
                            seed=seed,
                            protocol=protocol,
                            phase="development",
                            dataset=config.dataset,
                        )
                    )
            score = _development_score(runs)
            trials.append(
                {
                    "method": name,
                    "parameters": parameters,
                    "score": score,
                    "runs": runs,
                }
            )
            if score > best_score:
                best_score = score
                best_parameters = dict(parameters)
        if best_parameters is None:
            raise RuntimeError(f"no valid configurations for {name}")
        selected[name] = best_parameters
    return selected, trials


def _memory_matched_setup(
    config: SolverComparisonConfig,
    *,
    seed: int,
    selected: Mapping[str, Mapping[str, Any]],
    feature_map: OSELMFeatureMap,
    classes: int,
) -> _SolverSetup:
    full_width = config.hidden_size + 1
    input_size = feature_map.input_size
    fd = _base_setup(
        "fd_ridge",
        {**selected["fd_ridge"], "seed": seed},
        width=full_width,
        classes=classes,
    ).factory()
    target_bytes = fd.state_nbytes + _representation_bytes(full_width, input_size)
    candidates: list[tuple[int, int]] = []
    for width in range(2, full_width + 1):
        rls = RLSReadout(
            width,
            classes,
            seed=seed,
            regularization=float(selected["exact_rls"]["regularization"]),
            forgetting_factor=1.0,
        )
        total = rls.state_nbytes + _representation_bytes(width, input_size)
        candidates.append((abs(total - target_bytes), width))
    _, matched_width = min(candidates)
    bias_index = full_width - 1
    indices = tuple(range(matched_width - 1)) + (bias_index,)
    parameters = {
        "regularization": selected["exact_rls"]["regularization"],
        "feature_coordinates": matched_width,
        "matched_to": "fd_ridge_total_persistent_bytes",
        "seed": seed,
    }
    return _SolverSetup(
        factory=lambda: RLSReadout(
            matched_width,
            classes,
            seed=seed,
            regularization=float(parameters["regularization"]),
            forgetting_factor=1.0,
        ),
        adapter=FeatureSubsetAdapter(indices),
        parameters=parameters,
        representation_coordinates=matched_width,
    )


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "online_accuracy",
        "final_locked_accuracy",
        "sample_efficiency_auc",
        "mean_adaptation_delta",
        "average_forgetting",
        "backward_transfer",
    )
    resource_names = (
        "solver_state_bytes",
        "representation_state_bytes",
        "total_persistent_bytes",
        "update_mean_microseconds",
        "update_p95_microseconds",
        "samples_per_second",
        "peak_traced_training_bytes",
    )
    methods = sorted({run["method"] for run in runs})
    protocols = sorted({run["protocol"] for run in runs})
    summary: dict[str, Any] = {}
    for method in methods:
        summary[method] = {}
        for protocol in protocols:
            selected_runs = [
                run
                for run in runs
                if run["method"] == method and run["protocol"] == protocol
            ]
            metrics: dict[str, Any] = {}
            for name in metric_names:
                values = np.asarray(
                    [run["metrics"][name] for run in selected_runs],
                    dtype=np.float64,
                )
                metrics[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
            resources: dict[str, Any] = {}
            for name in resource_names:
                values = np.asarray(
                    [run["resources"][name] for run in selected_runs],
                    dtype=np.float64,
                )
                resources[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
            summary[method][protocol] = {
                "seeds": [run["seed"] for run in selected_runs],
                "metrics": metrics,
                "resources": resources,
            }
    return summary


def _findings(summary: Mapping[str, Any]) -> dict[str, Any]:
    quality: dict[str, float] = {}
    memory: dict[str, float] = {}
    solver_memory: dict[str, float] = {}
    representation_memory: dict[str, float] = {}
    compute: dict[str, float] = {}
    for method, protocols in summary.items():
        quality[method] = float(
            np.mean(
                [
                    result["metrics"]["final_locked_accuracy"]["mean"]
                    for result in protocols.values()
                ]
            )
        )
        memory[method] = float(
            np.mean(
                [
                    result["resources"]["total_persistent_bytes"]["mean"]
                    for result in protocols.values()
                ]
            )
        )
        solver_memory[method] = float(
            np.mean(
                [
                    result["resources"]["solver_state_bytes"]["mean"]
                    for result in protocols.values()
                ]
            )
        )
        representation_memory[method] = float(
            np.mean(
                [
                    result["resources"]["representation_state_bytes"]["mean"]
                    for result in protocols.values()
                ]
            )
        )
        compute[method] = float(
            np.mean(
                [
                    result["resources"]["update_mean_microseconds"]["mean"]
                    for result in protocols.values()
                ]
            )
        )
    predictive_winner = max(quality, key=quality.get)
    exact_solver_memory = solver_memory["exact_rls"]
    memory_budget = (
        representation_memory["exact_rls"] + 0.5 * exact_solver_memory
    )
    memory_eligible = {
        name: score
        for name, score in quality.items()
        if memory[name] <= memory_budget
    }
    compute_eligible = {
        name: score
        for name, score in quality.items()
        if compute[name] <= compute["exact_rls"]
    }
    return {
        "predictive_winner": predictive_winner,
        "predictive_winner_score": quality[predictive_winner],
        "memory_budget_definition": (
            "full frozen representation plus 50% of exact-RLS solver state"
        ),
        "memory_budget_bytes": memory_budget,
        "memory_budget_winner": (
            max(memory_eligible, key=memory_eligible.get)
            if memory_eligible
            else None
        ),
        "compute_budget_microseconds": compute["exact_rls"],
        "compute_budget_winner": (
            max(compute_eligible, key=compute_eligible.get)
            if compute_eligible
            else None
        ),
        "quality_scores": quality,
        "mean_total_persistent_bytes": memory,
        "mean_solver_state_bytes": solver_memory,
        "mean_update_microseconds": compute,
        "interpretation_limits": (
            "Winners are descriptive for held-out seeds and this benchmark; "
            "they are not evidence of universal dominance."
        ),
    }


def _paired_comparisons(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, protocols in summary.items():
        if method == "exact_rls":
            continue
        result[method] = {}
        for protocol, values in protocols.items():
            result[method][protocol] = {}
            exact = summary["exact_rls"][protocol]
            for metric in ("online_accuracy", "final_locked_accuracy"):
                candidate_values = np.asarray(
                    values["metrics"][metric]["values"], dtype=np.float64
                )
                exact_values = np.asarray(
                    exact["metrics"][metric]["values"], dtype=np.float64
                )
                differences = candidate_values - exact_values
                result[method][protocol][metric] = {
                    "mean_paired_delta": float(np.mean(differences)),
                    "std_paired_delta": (
                        float(np.std(differences, ddof=1))
                        if len(differences) > 1
                        else 0.0
                    ),
                    "wins": int(np.count_nonzero(differences > 0.0)),
                    "ties": int(np.count_nonzero(differences == 0.0)),
                    "losses": int(np.count_nonzero(differences < 0.0)),
                    "values": differences.tolist(),
                }
    return result


def run_solver_comparison(config: SolverComparisonConfig) -> dict[str, Any]:
    selected, development_trials = _tune(config)
    heldout_runs: list[dict[str, Any]] = []
    width = config.hidden_size + 1
    for seed in config.heldout_seeds:
        for protocol in ("shuffled_augmented", "class_ordered"):
            segments, evaluation_sets, feature_map = _materialize_problem(
                config, seed=seed, protocol=protocol, development=False
            )
            for name, parameters in selected.items():
                setup = _base_setup(
                    name,
                    {**parameters, "seed": seed},
                    width=width,
                    classes=len(evaluation_sets) - 1,
                )
                heldout_runs.append(
                    _run_setup(
                        name=name,
                        setup=setup,
                        segments=segments,
                        evaluation_sets=evaluation_sets,
                        feature_map=feature_map,
                        seed=seed,
                        protocol=protocol,
                        phase="heldout",
                        dataset=config.dataset,
                    )
                )
            heldout_runs.append(
                _run_setup(
                    name="memory_matched_exact_rls",
                    setup=_memory_matched_setup(
                        config,
                        seed=seed,
                        selected=selected,
                        feature_map=feature_map,
                        classes=len(evaluation_sets) - 1,
                    ),
                    segments=segments,
                    evaluation_sets=evaluation_sets,
                    feature_map=feature_map,
                    seed=seed,
                    protocol=protocol,
                    phase="heldout",
                    dataset=config.dataset,
                )
            )
    summary = _aggregate(heldout_runs)
    return {
        "schema_version": 1,
        "experiment": "os_elm_solver_comparison",
        "config": asdict(config),
        "protocol": {
            "representation": "fixed deterministic OS-ELM tanh features",
            "development_seeds_disjoint": True,
            "heldout_selected_once": True,
            "backpropagation": False,
            "raw_observation_replay": False,
        },
        "selected_hyperparameters": selected,
        "development_trials": development_trials,
        "heldout_runs": heldout_runs,
        "summary": summary,
        "paired_vs_exact_rls": _paired_comparisons(summary),
        "findings": _findings(summary),
    }


def run_fixed_solver_study(
    config: SolverComparisonConfig,
    selected_hyperparameters: Mapping[str, Mapping[str, Any]],
    *,
    methods: tuple[str, ...] = (
        "exact_rls",
        "fd_ridge",
        "memory_matched_exact_rls",
    ),
) -> dict[str, Any]:
    """Run a confirmatory study without inspecting or tuning held-out seeds."""

    required = {"exact_rls", "fd_ridge"}
    if not required <= set(selected_hyperparameters):
        raise ValueError("fixed studies require exact_rls and fd_ridge parameters")
    unknown = set(methods) - {
        "lms",
        "exact_rls",
        "diagonal_rls",
        "block_rls",
        "rpls",
        "fd_ridge",
        "memory_matched_exact_rls",
    }
    if unknown:
        raise ValueError("unknown fixed-study methods: " + ", ".join(sorted(unknown)))
    heldout_runs: list[dict[str, Any]] = []
    width = config.hidden_size + 1
    for seed in config.heldout_seeds:
        for protocol in ("shuffled_augmented", "class_ordered"):
            segments, evaluation_sets, feature_map = _materialize_problem(
                config, seed=seed, protocol=protocol, development=False
            )
            classes = len(evaluation_sets) - 1
            for name in methods:
                if name == "memory_matched_exact_rls":
                    setup = _memory_matched_setup(
                        config,
                        seed=seed,
                        selected=selected_hyperparameters,
                        feature_map=feature_map,
                        classes=classes,
                    )
                else:
                    if name not in selected_hyperparameters:
                        raise ValueError(f"no frozen parameters supplied for {name}")
                    setup = _base_setup(
                        name,
                        {**selected_hyperparameters[name], "seed": seed},
                        width=width,
                        classes=classes,
                    )
                heldout_runs.append(
                    _run_setup(
                        name=name,
                        setup=setup,
                        segments=segments,
                        evaluation_sets=evaluation_sets,
                        feature_map=feature_map,
                        seed=seed,
                        protocol=protocol,
                        phase="confirmatory",
                        dataset=config.dataset,
                    )
                )
    return combine_fixed_solver_runs(
        config,
        selected_hyperparameters,
        heldout_runs,
        methods=methods,
    )


def combine_fixed_solver_runs(
    config: SolverComparisonConfig,
    selected_hyperparameters: Mapping[str, Mapping[str, Any]],
    heldout_runs: list[dict[str, Any]],
    *,
    methods: tuple[str, ...] = (
        "exact_rls",
        "fd_ridge",
        "memory_matched_exact_rls",
    ),
) -> dict[str, Any]:
    """Combine independently executed seed runs without changing statistics."""

    if not heldout_runs:
        raise ValueError("at least one held-out run is required")
    required = {"exact_rls", "fd_ridge"}
    summary = _aggregate(heldout_runs)
    return {
        "schema_version": 1,
        "experiment": "os_elm_fixed_solver_study",
        "config": asdict(config),
        "protocol": {
            "representation": "fixed deterministic OS-ELM tanh features",
            "hyperparameters_frozen_before_heldout": True,
            "backpropagation": False,
            "raw_observation_replay": False,
        },
        "selected_hyperparameters": {
            name: dict(values)
            for name, values in selected_hyperparameters.items()
            if name in methods or name in required
        },
        "development_trials": [],
        "heldout_runs": heldout_runs,
        "summary": summary,
        "paired_vs_exact_rls": _paired_comparisons(summary),
        "findings": _findings(summary),
    }


def render_solver_report(result: Mapping[str, Any]) -> str:
    findings = result["findings"]
    dataset = result.get("config", {}).get("dataset", "digits")
    runs = result["heldout_runs"]
    heldout_seed_count = len({run["seed"] for run in runs})
    sample_counts = sorted({run["training"]["samples"] for run in runs})
    validation_failures = sum(
        int(
            run["numerical_stability"]["nonfinite_state_values"] != 0
            or not run["numerical_stability"]["bounded_state"]
            or not all(
                checkpoint["evaluation_sets"]["all"]["weights_unchanged"]
                for checkpoint in run["training"]["checkpoints"]
            )
        )
        for run in runs
    )
    lines = [
        "# OS-ELM output-solver comparison",
        "",
        f"Dataset: `{dataset}`. All methods use frozen OS-ELM features and no backpropagation.",
        "Hyperparameters were selected on disjoint development seeds or frozen before confirmatory evaluation.",
        "",
        "## Selected hyperparameters",
        "",
    ]
    for method, parameters in result["selected_hyperparameters"].items():
        lines.append(f"- `{method}`: `{parameters}`")
    lines.extend(
        [
            "",
            "## Held-out means",
            "",
            "Values are mean +/- sample standard deviation across held-out seeds.",
            "",
            "| Method | Protocol | Online acc. | Final locked acc. | Final delta vs exact | Forgetting | State bytes | Update us |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, protocols in result["summary"].items():
        for protocol, values in protocols.items():
            metrics = values["metrics"]
            resources = values["resources"]
            paired = (
                {"mean_paired_delta": 0.0, "std_paired_delta": 0.0}
                if method == "exact_rls"
                else result["paired_vs_exact_rls"][method][protocol][
                    "final_locked_accuracy"
                ]
            )
            lines.append(
                f"| {method} | {protocol} | "
                f"{metrics['online_accuracy']['mean']:.4f} +/- {metrics['online_accuracy']['std']:.4f} | "
                f"{metrics['final_locked_accuracy']['mean']:.4f} +/- {metrics['final_locked_accuracy']['std']:.4f} | "
                f"{paired['mean_paired_delta']:+.4f} +/- {paired['std_paired_delta']:.4f} | "
                f"{metrics['average_forgetting']['mean']:.4f} +/- {metrics['average_forgetting']['std']:.4f} | "
                f"{resources['total_persistent_bytes']['mean']:.0f} | "
                f"{resources['update_mean_microseconds']['mean']:.1f} +/- {resources['update_mean_microseconds']['std']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Secondary metrics",
            "",
            "| Method | Protocol | Sample-efficiency AUC | Adaptation delta | Backward transfer | Samples/s | Peak traced bytes |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, protocols in result["summary"].items():
        for protocol, values in protocols.items():
            metrics = values["metrics"]
            resources = values["resources"]
            lines.append(
                f"| {method} | {protocol} | "
                f"{metrics['sample_efficiency_auc']['mean']:.4f} +/- {metrics['sample_efficiency_auc']['std']:.4f} | "
                f"{metrics['mean_adaptation_delta']['mean']:+.4f} +/- {metrics['mean_adaptation_delta']['std']:.4f} | "
                f"{metrics['backward_transfer']['mean']:+.4f} +/- {metrics['backward_transfer']['std']:.4f} | "
                f"{resources['samples_per_second']['mean']:.1f} +/- {resources['samples_per_second']['std']:.1f} | "
                f"{resources['peak_traced_training_bytes']['mean']:.0f} +/- {resources['peak_traced_training_bytes']['std']:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Descriptive winners",
            "",
            f"- Predictive quality: `{findings['predictive_winner']}`.",
            f"- Under the declared reduced-state budget: `{findings['memory_budget_winner']}`.",
            f"- At or below exact-RLS update time: `{findings['compute_budget_winner']}`.",
            "",
            "## Interpretation",
            "",
            "## Validation audit",
            "",
            f"The study contains {len(runs)} runs over {heldout_seed_count} held-out seeds. Per-run update counts were {sample_counts}. Validation failures: {validation_failures}.",
            "",
            "These winners are benchmark-specific descriptive results, not a "
            "claim of universal or statistically definitive dominance. Raw "
            "development and held-out runs are retained alongside this report.",
            "",
            "Peak memory is reported as traced Python/NumPy allocations. "
            "Per-method isolated peak process RSS was not reliably measurable "
            "inside this single-process paired runner and is recorded as null.",
            "",
        ]
    )
    interpretation_index = lines.index("## Validation audit") - 1
    interpretations: list[str] = []
    quality = findings["quality_scores"]
    memory = findings["mean_total_persistent_bytes"]
    compute = findings["mean_update_microseconds"]
    if "exact_rls" in quality:
        interpretations.append(
            f"- Exact RLS mean final score across protocols: {quality['exact_rls']:.4f}."
        )
    if "memory_matched_exact_rls" in quality and "exact_rls" in quality:
        memory_reduction = 1.0 - memory["memory_matched_exact_rls"] / memory["exact_rls"]
        accuracy_delta = quality["memory_matched_exact_rls"] - quality["exact_rls"]
        interpretations.append(
            f"- Memory-matched smaller exact RLS changed final accuracy by {accuracy_delta:+.4f} while changing total persistent state by {-memory_reduction:+.1%}."
        )
    if "fd_ridge" in quality and "exact_rls" in quality:
        interpretations.append(
            f"- Frequent-Directions ridge changed final accuracy by {quality['fd_ridge'] - quality['exact_rls']:+.4f}, used {memory['fd_ridge']} bytes, and took {compute['fd_ridge'] / compute['exact_rls']:.1f}x the exact-RLS update time."
        )
    if "rpls" in quality and "exact_rls" in quality:
        interpretations.append(
            f"- RPLS changed final accuracy by {quality['rpls'] - quality['exact_rls']:+.4f} and took {compute['rpls'] / compute['exact_rls']:.1f}x the exact-RLS update time."
        )
    lines[interpretation_index:interpretation_index] = interpretations
    return "\n".join(lines)


def write_solver_artifacts(
    result: Mapping[str, Any], destination: str | Path
) -> Path:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    write_json_result(result, root / "comparison.json")
    for run in result["heldout_runs"]:
        filename = f"{run['protocol']}__{run['method']}__seed_{run['seed']}.json"
        write_json_result(run, root / "raw" / filename)
    for index, trial in enumerate(result["development_trials"]):
        write_json_result(trial, root / "development" / f"trial_{index:03d}.json")
    (root / "REPORT.md").write_text(
        render_solver_report(result), encoding="utf-8"
    )
    return root

"""Capstone experiment for bounded associative memory under recurring regimes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from no_backprop.digits import DigitsSplit, load_digits_split
from no_backprop.drift_suite import DriftSuiteConfig, transform_images
from no_backprop.experiment import _evaluate_digits_locked
from no_backprop.milestone6 import _train_phase
from no_backprop.readouts import (
    CumulativeMaturityReadout,
    ManagedProbationaryMaturityReadout,
    RLSReadout,
)
from no_backprop.spatial import OnlineSpatialClassifier, SpatialClassifierConfig


RegimeName = Literal[
    "original",
    "inversion",
    "translation",
    "center_occlusion",
]


@dataclass(frozen=True)
class MemoryCapstoneConfig:
    """Matched recurring regimes, permanent capacities, and RLS discounts."""

    seeds: tuple[int, ...] = (3, 7, 11, 17, 23, 29, 37, 41, 47, 53)
    test_per_class: int = 40
    phase_domains: tuple[RegimeName, ...] = (
        "original",
        "inversion",
        "translation",
        "center_occlusion",
        "original",
        "inversion",
    )
    mature_capacities: tuple[int, ...] = (8, 16, 32, 64)
    candidate_capacity: int = 16
    forgetting_factors: tuple[float, ...] = (
        1.0,
        0.99999,
        0.9999,
        0.999,
        0.995,
        0.99,
        0.98,
        0.95,
    )
    regularization: float = 1.0
    rbf_width: float = 0.05
    min_center_distance: float = 0.01
    translation_pixels: int = 1
    occlusion_size: int = 2

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if self.test_per_class <= 0:
            raise ValueError("test_per_class must be positive")
        allowed = set(RegimeName.__args__)
        if not self.phase_domains or any(
            domain not in allowed for domain in self.phase_domains
        ):
            raise ValueError("phase_domains contain an unknown regime")
        if self.phase_domains.count("original") < 2:
            raise ValueError("original must recur at least once")
        if self.phase_domains.count("inversion") < 2:
            raise ValueError("inversion must recur at least once")
        if (
            not self.mature_capacities
            or len(set(self.mature_capacities)) != len(self.mature_capacities)
            or any(capacity <= 0 for capacity in self.mature_capacities)
        ):
            raise ValueError("mature_capacities must be positive and unique")
        if self.candidate_capacity <= 0:
            raise ValueError("candidate_capacity must be positive")
        if (
            not self.forgetting_factors
            or len(set(self.forgetting_factors))
            != len(self.forgetting_factors)
            or 1.0 not in self.forgetting_factors
            or any(not 0.0 < factor <= 1.0 for factor in self.forgetting_factors)
        ):
            raise ValueError(
                "forgetting_factors must be unique, in (0, 1], and include 1"
            )
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.rbf_width <= 0.0:
            raise ValueError("rbf_width must be positive")
        if self.min_center_distance < 0.0:
            raise ValueError("min_center_distance cannot be negative")


def _factor_name(factor: float) -> str:
    return f"rls_ff_{factor:g}"


def _managed_name(capacity: int) -> str:
    return f"managed_memory_{capacity}"


def _model_names(config: MemoryCapstoneConfig) -> tuple[str, ...]:
    return (
        *(_factor_name(factor) for factor in config.forgetting_factors),
        "immediate_maturity_32",
        *(_managed_name(capacity) for capacity in config.mature_capacities),
    )


def _build_learner(
    name: str, config: MemoryCapstoneConfig, seed: int
) -> OnlineSpatialClassifier:
    input_size = 65
    output_size = 10
    if name.startswith("rls_ff_"):
        factor = float(name.removeprefix("rls_ff_"))
        readout = RLSReadout(
            input_size,
            output_size,
            seed=seed,
            regularization=config.regularization,
            forgetting_factor=factor,
        )
    elif name == "immediate_maturity_32":
        readout = CumulativeMaturityReadout(
            input_size,
            output_size,
            seed=seed,
            regularization=config.regularization,
            max_neurons=32,
            rbf_width=config.rbf_width,
            min_center_distance=config.min_center_distance,
            leverage_gated=True,
        )
    elif name.startswith("managed_memory_"):
        capacity = int(name.removeprefix("managed_memory_"))
        readout = ManagedProbationaryMaturityReadout(
            input_size,
            output_size,
            seed=seed,
            regularization=config.regularization,
            max_neurons=capacity,
            rbf_width=config.rbf_width,
            min_center_distance=config.min_center_distance,
            max_candidates=config.candidate_capacity,
        )
    else:  # pragma: no cover - names are generated internally
        raise ValueError(f"unknown capstone model: {name}")
    return OnlineSpatialClassifier(
        SpatialClassifierConfig(
            image_size=8,
            output_size=output_size,
            frontend="signed_magnitude_convolution",
            seed=seed,
        ),
        readout,
    )


def _regime_splits(
    split: DigitsSplit, config: MemoryCapstoneConfig, seed: int
) -> dict[RegimeName, DigitsSplit]:
    drift_config = DriftSuiteConfig(
        seeds=(seed,),
        test_per_class=config.test_per_class,
        transformations=("inversion", "translation", "center_occlusion"),
        translation_pixels=config.translation_pixels,
        occlusion_size=config.occlusion_size,
    )
    regimes: dict[RegimeName, DigitsSplit] = {"original": split}
    for index, name in enumerate(
        ("inversion", "translation", "center_occlusion")
    ):
        regimes[name] = DigitsSplit(
            train_images=transform_images(
                split.train_images,
                name,
                drift_config,
                seed=seed + 1_000 * (index + 1),
            ),
            train_labels=split.train_labels.copy(),
            test_images=transform_images(
                split.test_images,
                name,
                drift_config,
                seed=seed + 1_000 * (index + 1) + 1,
            ),
            test_labels=split.test_labels.copy(),
        )
    return regimes


def _memory_phase_before(
    learner: OnlineSpatialClassifier,
) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    readout = learner.readout
    if not isinstance(readout, CumulativeMaturityReadout):
        return 0, None, None
    count = int(readout.active_count[0])
    return (
        count,
        readout.neuron_centers[:count].copy(),
        readout.neuron_evidence[:count].copy(),
    )


def _memory_phase_after(
    learner: OnlineSpatialClassifier,
    existing_count: int,
    centers_before: np.ndarray | None,
    evidence_before: np.ndarray | None,
) -> dict[str, float | int | bool | None]:
    readout = learner.readout
    if not isinstance(readout, CumulativeMaturityReadout):
        return {
            "mature_capacity": None,
            "active_neurons": None,
            "new_mature_neurons": None,
            "existing_neurons_reactivated": None,
            "existing_neuron_evidence_gain": None,
            "maximum_existing_center_shift": None,
            "capacity_full": False,
        }
    count = int(readout.active_count[0])
    evidence_gain = readout.neuron_evidence[:existing_count] - evidence_before
    if existing_count == 0:
        maximum_shift = 0.0
        reactivated = 0
        total_evidence_gain = 0.0
    else:
        maximum_shift = float(
            np.max(
                np.abs(
                    readout.neuron_centers[:existing_count] - centers_before
                )
            )
        )
        reactivated = int(np.sum(evidence_gain > 1e-6))
        total_evidence_gain = float(np.sum(evidence_gain))
    return {
        "mature_capacity": readout.max_neurons,
        "active_neurons": count,
        "new_mature_neurons": count - existing_count,
        "existing_neurons_reactivated": reactivated,
        "existing_neuron_evidence_gain": total_evidence_gain,
        "maximum_existing_center_shift": maximum_shift,
        "capacity_full": count == readout.max_neurons,
    }


def _run_model(
    name: str,
    config: MemoryCapstoneConfig,
    regimes: dict[RegimeName, DigitsSplit],
    *,
    seed: int,
) -> dict[str, Any]:
    learner = _build_learner(name, config, seed)
    state_before = learner.state_nbytes
    seen: set[RegimeName] = set()
    occurrences: dict[RegimeName, int] = {}
    phases: list[dict[str, Any]] = []
    trained_samples = 0
    training_seconds = 0.0
    capacity_filled_phase: int | None = None
    for phase_index, domain in enumerate(config.phase_domains):
        split = regimes[domain]
        occurrence = occurrences.get(domain, 0) + 1
        occurrences[domain] = occurrence
        pre_evaluation = _evaluate_digits_locked(
            learner, split.test_images, split.test_labels
        )
        existing, centers_before, evidence_before = _memory_phase_before(
            learner
        )
        started = perf_counter()
        training = _train_phase(
            learner,
            split,
            seed=seed + 100 * (phase_index + 1),
        )
        elapsed = perf_counter() - started
        trained_samples += int(training["samples"])
        training_seconds += elapsed
        memory = _memory_phase_after(
            learner, existing, centers_before, evidence_before
        )
        if memory["capacity_full"] and capacity_filled_phase is None:
            capacity_filled_phase = phase_index + 1
        seen.add(domain)
        evaluations = {
            evaluated_domain: _evaluate_digits_locked(
                learner,
                evaluated_split.test_images,
                evaluated_split.test_labels,
            )
            for evaluated_domain, evaluated_split in regimes.items()
        }
        phases.append(
            {
                "phase_index": phase_index + 1,
                "domain": domain,
                "occurrence": occurrence,
                "pre_domain_accuracy": pre_evaluation["accuracy"],
                "pre_evaluation_locked": pre_evaluation["weights_unchanged"]
                and pre_evaluation["transient_state_restored"],
                "training": training,
                "training_seconds": elapsed,
                "post_domain_accuracy": evaluations[domain]["accuracy"],
                "mean_seen_domain_accuracy": float(
                    np.mean(
                        [evaluations[name]["accuracy"] for name in seen]
                    )
                ),
                "evaluations": evaluations,
                "memory": memory,
            }
        )
    final_evaluations = phases[-1]["evaluations"]
    return {
        "model": name,
        "phases": phases,
        "state_bytes_before": state_before,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": state_before == learner.state_nbytes,
        "trained_samples": trained_samples,
        "training_images_per_second": trained_samples / training_seconds,
        "final_mean_domain_accuracy": float(
            np.mean(
                [evaluation["accuracy"] for evaluation in final_evaluations.values()]
            )
        ),
        "capacity_filled_phase": capacity_filled_phase,
        "readout_diagnostics": (
            learner.readout.diagnostics
            if isinstance(learner.readout, CumulativeMaturityReadout)
            else {
                "forgetting_factor": learner.readout.forgetting_factor,
                "stored_raw_samples": 0,
            }
        ),
    }


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": values,
    }


def _phase_metrics(model: dict[str, Any]) -> dict[str, float]:
    phases = model["phases"]
    first_shifted = [
        phase
        for phase in phases
        if phase["occurrence"] == 1 and phase["domain"] != "original"
    ]
    returns = [phase for phase in phases if phase["occurrence"] > 1]
    return {
        "mean_online_accuracy": float(
            np.mean([phase["training"]["online_accuracy"] for phase in phases])
        ),
        "first_shifted_online_accuracy": float(
            np.mean(
                [phase["training"]["online_accuracy"] for phase in first_shifted]
            )
        ),
        "mean_return_pre_accuracy": float(
            np.mean([phase["pre_domain_accuracy"] for phase in returns])
        ),
        "mean_return_online_accuracy": float(
            np.mean([phase["training"]["online_accuracy"] for phase in returns])
        ),
        "mean_return_post_accuracy": float(
            np.mean([phase["post_domain_accuracy"] for phase in returns])
        ),
        "mean_seen_domain_accuracy": float(
            np.mean([phase["mean_seen_domain_accuracy"] for phase in phases])
        ),
        "final_mean_domain_accuracy": float(model["final_mean_domain_accuracy"]),
        "training_images_per_second": float(
            model["training_images_per_second"]
        ),
        "state_bytes_after": float(model["state_bytes_after"]),
    }


def _return_metrics(
    model: dict[str, Any], domain: RegimeName
) -> dict[str, float]:
    appearances = [
        phase for phase in model["phases"] if phase["domain"] == domain
    ]
    first, returned = appearances[0], appearances[-1]
    return {
        "retention_before_return": float(
            returned["pre_domain_accuracy"] - first["post_domain_accuracy"]
        ),
        "return_pre_accuracy": float(returned["pre_domain_accuracy"]),
        "online_return_savings": float(
            returned["training"]["online_accuracy"]
            - first["training"]["online_accuracy"]
        ),
        "return_online_accuracy": float(
            returned["training"]["online_accuracy"]
        ),
        "return_reacquisition_gain": float(
            returned["post_domain_accuracy"] - returned["pre_domain_accuracy"]
        ),
        "return_post_accuracy": float(returned["post_domain_accuracy"]),
        "existing_neurons_reactivated": float(
            returned["memory"]["existing_neurons_reactivated"] or 0
        ),
        "existing_neuron_evidence_gain": float(
            returned["memory"]["existing_neuron_evidence_gain"] or 0.0
        ),
    }


def _capacity_metrics(model: dict[str, Any]) -> dict[str, float]:
    phases = model["phases"]
    full_phase = model["capacity_filled_phase"]
    active_after = float(phases[-1]["memory"]["active_neurons"] or 0)
    capacity = float(phases[-1]["memory"]["mature_capacity"] or 0)
    saturated = (
        phases if full_phase is None else phases[int(full_phase) - 1 :]
    )
    return {
        "capacity_filled_by_end": float(full_phase is not None),
        "first_full_phase_or_end_plus_one": float(
            len(phases) + 1 if full_phase is None else full_phase
        ),
        "final_active_neurons": active_after,
        "final_capacity_utilization": active_after / capacity,
        "mean_online_accuracy_from_full_phase_or_start_if_unfilled": float(
            np.mean(
                [phase["training"]["online_accuracy"] for phase in saturated]
            )
        ),
    }


def run_memory_capstone(
    config: MemoryCapstoneConfig = MemoryCapstoneConfig(),
) -> dict[str, Any]:
    """Run recurring regimes with matched cumulative and discounted learners."""

    names = _model_names(config)
    runs: list[dict[str, Any]] = []
    labels_preserved = True
    for seed in config.seeds:
        original = load_digits_split(
            test_per_class=config.test_per_class, seed=seed
        )
        regimes = _regime_splits(original, config, seed)
        labels_preserved = labels_preserved and all(
            np.array_equal(split.train_labels, original.train_labels)
            and np.array_equal(split.test_labels, original.test_labels)
            for split in regimes.values()
        )
        runs.append(
            {
                "seed": seed,
                "models": {
                    name: _run_model(name, config, regimes, seed=seed)
                    for name in names
                },
            }
        )

    metric_runs = [
        {name: _phase_metrics(run["models"][name]) for name in names}
        for run in runs
    ]
    overall_metrics = tuple(next(iter(metric_runs))[names[0]])
    overall = {
        name: {
            metric: _summary(
                [float(run[name][metric]) for run in metric_runs]
            )
            for metric in overall_metrics
        }
        for name in names
    }
    baseline = _factor_name(1.0)
    paired = {
        name: {
            metric: _summary(
                [
                    float(run[name][metric]) - float(run[baseline][metric])
                    for run in metric_runs
                ]
            )
            for metric in overall_metrics
        }
        for name in names
        if name != baseline
    }
    recurring_domains: tuple[RegimeName, ...] = ("original", "inversion")
    returns = {
        domain: {
            name: {
                metric: _summary(
                    [
                        _return_metrics(run["models"][name], domain)[metric]
                        for run in runs
                    ]
                )
                for metric in _return_metrics(
                    runs[0]["models"][name], domain
                )
            }
            for name in names
        }
        for domain in recurring_domains
    }
    phase_summary = {
        str(index + 1): {
            "domain": domain,
            "models": {
                name: {
                    metric: _summary(
                        [
                            float(run["models"][name]["phases"][index][metric])
                            for run in runs
                        ]
                    )
                    for metric in (
                        "pre_domain_accuracy",
                        "post_domain_accuracy",
                        "mean_seen_domain_accuracy",
                    )
                }
                | {
                    "online_accuracy": _summary(
                        [
                            float(
                                run["models"][name]["phases"][index][
                                    "training"
                                ]["online_accuracy"]
                            )
                            for run in runs
                        ]
                    ),
                    "active_neurons": _summary(
                        [
                            float(
                                run["models"][name]["phases"][index]["memory"][
                                    "active_neurons"
                                ]
                                or 0
                            )
                            for run in runs
                        ]
                    ),
                }
                for name in names
            },
        }
        for index, domain in enumerate(config.phase_domains)
    }
    capacity_summary = {
        name: {
            metric: _summary(
                [
                    _capacity_metrics(run["models"][name])[metric]
                    for run in runs
                ]
            )
            for metric in _capacity_metrics(runs[0]["models"][name])
        }
        for name in names
        if name.startswith("managed_memory_")
    }
    model_runs = [model for run in runs for model in run["models"].values()]
    phase_runs = [phase for model in model_runs for phase in model["phases"]]
    return {
        "experiment": "bounded_associative_memory_capstone",
        "config": asdict(config),
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "downloaded_data_bytes": 0,
            "regimes": sorted(set(config.phase_domains)),
            "labels_preserved_by_regimes": labels_preserved,
        },
        "baseline": baseline,
        "models": list(names),
        "invariants": {
            "same_fixed_signed_magnitude_frontend": True,
            "one_label_update_per_training_image": all(
                model["trained_samples"]
                == len(config.phase_domains)
                * model["phases"][0]["training"]["samples"]
                for model in model_runs
            ),
            "weights_locked_during_evaluation": all(
                phase["pre_evaluation_locked"]
                and all(
                    evaluation["weights_unchanged"]
                    and evaluation["transient_state_restored"]
                    for evaluation in phase["evaluations"].values()
                )
                for phase in phase_runs
            ),
            "mature_centers_are_frozen": all(
                phase["memory"]["maximum_existing_center_shift"] in (None, 0.0)
                for phase in phase_runs
            ),
            "bounded_state": all(model["bounded_state"] for model in model_runs),
            "raw_samples_stored_by_models": 0,
            "rls_factor_one_discards_historical_weight": False,
        },
        "runs": runs,
        "summary": {
            "overall": overall,
            "paired_difference_from_rls_ff_1": paired,
            "returns": returns,
            "phases": phase_summary,
            "factor_frontier": {
                str(factor): {
                    "model": _factor_name(factor),
                    "approximate_effective_history": (
                        None if factor == 1.0 else 1.0 / (1.0 - factor)
                    ),
                    "metrics": overall[_factor_name(factor)],
                }
                for factor in config.forgetting_factors
            },
            "capacity_frontier": {
                str(capacity): {
                    "model": _managed_name(capacity),
                    "metrics": overall[_managed_name(capacity)],
                    "saturation": capacity_summary[
                        _managed_name(capacity)
                    ],
                }
                for capacity in config.mature_capacities
            },
        },
    }

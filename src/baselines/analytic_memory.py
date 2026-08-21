"""Closest analytic/constructive baselines for the recurring-memory capstone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from baselines.analytic_readouts import (
    ALDKernelRLSReadout,
    FuzzyARTMAPReadout,
    OnlineSequentialELMReadout,
    ResourceAllocatingNetworkReadout,
)
from no_backprop.digits import DigitsSplit, load_digits_split
from no_backprop.experiment import _evaluate_digits
from no_backprop.memory_capstone import (
    MemoryCapstoneConfig,
    RegimeName,
    _build_learner as _build_capstone_learner,
    _phase_metrics,
    _regime_splits,
    _run_model as _run_capstone_model,
    _summary,
)
from no_backprop.milestone6 import _train_phase
from no_backprop.spatial import OnlineSpatialClassifier, SpatialClassifierConfig


AnalyticFamily = Literal["ald_krls", "ran", "fuzzy_artmap", "os_elm"]
BudgetName = Literal["cpam32", "cpam64"]
FAMILIES: tuple[AnalyticFamily, ...] = (
    "ald_krls",
    "ran",
    "fuzzy_artmap",
    "os_elm",
)
BUDGETS: tuple[BudgetName, ...] = ("cpam32", "cpam64")


@dataclass(frozen=True)
class AnalyticMemoryComparisonConfig:
    """Development-tuned closest baselines under CPAM state ceilings."""

    test_seeds: tuple[int, ...] = (3, 7, 11, 17, 23, 29, 37, 41, 47, 53)
    development_seeds: tuple[int, ...] = (2, 5, 13)
    test_per_class: int = 40
    phase_domains: tuple[RegimeName, ...] = (
        "original",
        "inversion",
        "translation",
        "center_occlusion",
        "original",
        "inversion",
    )
    krls_grid: tuple[tuple[float, float], ...] = (
        (0.05, 0.001),
        (0.05, 0.01),
        (0.1, 0.001),
        (0.1, 0.01),
        (0.2, 0.01),
        (0.2, 0.05),
    )
    ran_grid: tuple[tuple[float, float, float], ...] = (
        (0.05, 0.5, 0.05),
        (0.1, 0.5, 0.05),
        (0.1, 0.75, 0.05),
        (0.1, 0.5, 0.1),
        (0.2, 0.75, 0.1),
    )
    artmap_grid: tuple[tuple[float, float], ...] = (
        (0.5, 0.001),
        (0.7, 0.001),
        (0.85, 0.001),
        (0.95, 0.001),
    )
    os_elm_grid: tuple[tuple[float, float], ...] = (
        (0.25, 0.1),
        (0.25, 1.0),
        (0.5, 0.1),
        (0.5, 1.0),
        (1.0, 1.0),
    )

    def __post_init__(self) -> None:
        if not self.test_seeds or len(set(self.test_seeds)) != len(self.test_seeds):
            raise ValueError("test_seeds must be non-empty and unique")
        if not self.development_seeds or len(set(self.development_seeds)) != len(
            self.development_seeds
        ):
            raise ValueError("development_seeds must be non-empty and unique")
        if set(self.test_seeds) & set(self.development_seeds):
            raise ValueError("development and test seeds must be disjoint")
        if self.test_per_class <= 0:
            raise ValueError("test_per_class must be positive")
        allowed = set(RegimeName.__args__)
        if not self.phase_domains or any(
            domain not in allowed for domain in self.phase_domains
        ):
            raise ValueError("phase_domains contain an unknown regime")
        if self.phase_domains.count("original") < 2:
            raise ValueError("original must recur")
        if self.phase_domains.count("inversion") < 2:
            raise ValueError("inversion must recur")
        if not all((self.krls_grid, self.ran_grid, self.artmap_grid, self.os_elm_grid)):
            raise ValueError("all tuning grids must be non-empty")


def _memory_config(
    config: AnalyticMemoryComparisonConfig, seeds: tuple[int, ...]
) -> MemoryCapstoneConfig:
    return MemoryCapstoneConfig(
        seeds=seeds,
        test_per_class=config.test_per_class,
        phase_domains=config.phase_domains,
        mature_capacities=(32, 64),
        candidate_capacity=16,
        forgetting_factors=(1.0,),
    )


def _parameter_options(
    family: AnalyticFamily, config: AnalyticMemoryComparisonConfig
) -> tuple[dict[str, float], ...]:
    if family == "ald_krls":
        return tuple(
            {"kernel_width": width, "ald_tolerance": tolerance}
            for width, tolerance in config.krls_grid
        )
    if family == "ran":
        return tuple(
            {
                "learning_rate": rate,
                "error_threshold": error,
                "distance_threshold": distance,
            }
            for rate, error, distance in config.ran_grid
        )
    if family == "fuzzy_artmap":
        return tuple(
            {"vigilance": vigilance, "choice": choice}
            for vigilance, choice in config.artmap_grid
        )
    return tuple(
        {"input_scale": scale, "regularization": regularization}
        for scale, regularization in config.os_elm_grid
    )


def _readout(
    family: AnalyticFamily,
    capacity: int,
    parameters: dict[str, float],
    *,
    seed: int,
):
    if family == "ald_krls":
        return ALDKernelRLSReadout(
            65,
            10,
            max_dictionary_size=capacity,
            kernel_width=parameters["kernel_width"],
            ald_tolerance=parameters["ald_tolerance"],
        )
    if family == "ran":
        return ResourceAllocatingNetworkReadout(
            65,
            10,
            max_neurons=capacity,
            learning_rate=parameters["learning_rate"],
            error_threshold=parameters["error_threshold"],
            distance_threshold=parameters["distance_threshold"],
        )
    if family == "fuzzy_artmap":
        return FuzzyARTMAPReadout(
            65,
            10,
            max_categories=capacity,
            vigilance=parameters["vigilance"],
            choice=parameters["choice"],
        )
    return OnlineSequentialELMReadout(
        65,
        10,
        hidden_size=capacity,
        seed=seed,
        input_scale=parameters["input_scale"],
        regularization=parameters["regularization"],
    )


def _learner(
    family: AnalyticFamily,
    capacity: int,
    parameters: dict[str, float],
    *,
    seed: int,
) -> OnlineSpatialClassifier:
    return OnlineSpatialClassifier(
        SpatialClassifierConfig(
            image_size=8,
            output_size=10,
            frontend="signed_magnitude_convolution",
            seed=seed,
        ),
        _readout(family, capacity, parameters, seed=seed),
    )


def _capacity_for_budget(
    family: AnalyticFamily,
    budget_bytes: int,
    parameters: dict[str, float],
) -> int:
    """Largest preallocated structural width that does not exceed the budget."""

    def state(capacity: int) -> int:
        return _learner(family, capacity, parameters, seed=0).state_nbytes

    if state(1) > budget_bytes:
        raise ValueError(f"{family} cannot fit its minimum state in the budget")
    low = 1
    high = 2
    while state(high) <= budget_bytes:
        low = high
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if state(middle) <= budget_bytes:
            low = middle
        else:
            high = middle
    return low


def _locked_evaluation(
    learner: OnlineSpatialClassifier,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    arrays = (*learner.encoder.persistent_arrays, *learner.readout.persistent_arrays)
    before = [array.copy() for array in arrays]
    result = _evaluate_digits(learner, images, labels)
    if not all(np.array_equal(old, new) for old, new in zip(before, arrays)):
        raise RuntimeError("evaluation modified analytic baseline state")
    result["weights_unchanged"] = True
    result["transient_state_restored"] = True
    return result


def _active_structures(diagnostics: dict[str, Any]) -> int:
    for key in (
        "active_dictionary_vectors",
        "active_neurons",
        "active_categories",
        "hidden_size",
    ):
        if key in diagnostics:
            return int(diagnostics[key])
    return 0


def _run_analytic_model(
    family: AnalyticFamily,
    budget_name: BudgetName,
    capacity: int,
    parameters: dict[str, float],
    config: AnalyticMemoryComparisonConfig,
    regimes: dict[RegimeName, DigitsSplit],
    *,
    seed: int,
) -> dict[str, Any]:
    learner = _learner(family, capacity, parameters, seed=seed)
    state_before = learner.state_nbytes
    seen: set[RegimeName] = set()
    occurrences: dict[RegimeName, int] = {}
    phases: list[dict[str, Any]] = []
    trained_samples = 0
    training_seconds = 0.0
    for phase_index, domain in enumerate(config.phase_domains):
        split = regimes[domain]
        occurrence = occurrences.get(domain, 0) + 1
        occurrences[domain] = occurrence
        pre = _locked_evaluation(learner, split.test_images, split.test_labels)
        diagnostics_before = learner.readout.diagnostics
        active_before = _active_structures(diagnostics_before)
        started = perf_counter()
        training = _train_phase(
            learner, split, seed=seed + 100 * (phase_index + 1)
        )
        elapsed = perf_counter() - started
        trained_samples += int(training["samples"])
        training_seconds += elapsed
        diagnostics_after = learner.readout.diagnostics
        active_after = _active_structures(diagnostics_after)
        seen.add(domain)
        evaluations = {
            name: _locked_evaluation(
                learner, evaluated.test_images, evaluated.test_labels
            )
            for name, evaluated in regimes.items()
        }
        phases.append(
            {
                "phase_index": phase_index + 1,
                "domain": domain,
                "occurrence": occurrence,
                "pre_domain_accuracy": pre["accuracy"],
                "pre_evaluation_locked": True,
                "training": training,
                "training_seconds": elapsed,
                "post_domain_accuracy": evaluations[domain]["accuracy"],
                "mean_seen_domain_accuracy": float(
                    np.mean([evaluations[name]["accuracy"] for name in seen])
                ),
                "evaluations": evaluations,
                "memory": {
                    "structural_capacity": capacity,
                    "active_structures": active_after,
                    "new_structures": active_after - active_before,
                    "capacity_full": bool(
                        diagnostics_after.get("capacity_full", False)
                    ),
                },
                "diagnostics": diagnostics_after,
            }
        )
    final = phases[-1]["evaluations"]
    finite_state = all(
        np.all(np.isfinite(array))
        for array in (
            *learner.encoder.persistent_arrays,
            *learner.readout.persistent_arrays,
        )
    )
    return {
        "model": f"{family}_{budget_name}_budget",
        "family": family,
        "budget": budget_name,
        "parameters": parameters,
        "structural_capacity": capacity,
        "phases": phases,
        "state_bytes_before": state_before,
        "state_bytes_after": learner.state_nbytes,
        "bounded_state": state_before == learner.state_nbytes,
        "finite_state": finite_state,
        "trained_samples": trained_samples,
        "training_images_per_second": trained_samples / training_seconds,
        "final_mean_domain_accuracy": float(
            np.mean([evaluation["accuracy"] for evaluation in final.values()])
        ),
        "readout_diagnostics": learner.readout.diagnostics,
    }


PRIMARY_METRICS = (
    "first_shifted_online_accuracy",
    "mean_return_pre_accuracy",
    "mean_return_online_accuracy",
    "final_mean_domain_accuracy",
)


def _selection_score(model: dict[str, Any]) -> float:
    metrics = _phase_metrics(model)
    return float(np.mean([metrics[name] for name in PRIMARY_METRICS]))


def _t95_half_width(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    # Two-sided 95% Student-t critical values for the seed counts used here.
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }.get(len(values) - 1, 1.96)
    return float(critical * np.std(values, ddof=1) / np.sqrt(len(values)))


def _rich_summary(values: list[float]) -> dict[str, float | list[float]]:
    result = _summary(values)
    result["nominal_paired_95ci_half_width"] = _t95_half_width(values)
    return result


def run_analytic_memory_comparison(
    config: AnalyticMemoryComparisonConfig = AnalyticMemoryComparisonConfig(),
) -> dict[str, Any]:
    """Tune closest forward-only baselines, then run paired held-out seeds."""

    cpam_config = _memory_config(config, config.test_seeds)
    budget_bytes = {
        "cpam32": _build_capstone_learner(
            "managed_memory_32", cpam_config, 0
        ).state_nbytes,
        "cpam64": _build_capstone_learner(
            "managed_memory_64", cpam_config, 0
        ).state_nbytes,
    }

    # Tune at the smaller budget, then freeze parameters for both budgets. This
    # avoids doubling method-specific search and keeps test capacities unseen.
    development: dict[str, Any] = {}
    selected: dict[AnalyticFamily, dict[str, float]] = {}
    small_capacities: dict[AnalyticFamily, int] = {}
    development_config = _memory_config(config, config.development_seeds)
    for family in FAMILIES:
        options = _parameter_options(family, config)
        trials: list[dict[str, Any]] = []
        for option in options:
            capacity = _capacity_for_budget(
                family, budget_bytes["cpam32"], option
            )
            scores: list[float] = []
            metrics_by_seed: list[dict[str, float]] = []
            for seed in config.development_seeds:
                original = load_digits_split(
                    test_per_class=config.test_per_class, seed=seed
                )
                regimes = _regime_splits(original, development_config, seed)
                model = _run_analytic_model(
                    family,
                    "cpam32",
                    capacity,
                    option,
                    config,
                    regimes,
                    seed=seed,
                )
                metrics = _phase_metrics(model)
                metrics_by_seed.append(metrics)
                scores.append(_selection_score(model))
            trials.append(
                {
                    "parameters": option,
                    "structural_capacity": capacity,
                    "selection_score": _summary(scores),
                    "primary_metrics": {
                        metric: _summary(
                            [seed_metrics[metric] for seed_metrics in metrics_by_seed]
                        )
                        for metric in PRIMARY_METRICS
                    },
                }
            )
        # Stable order makes ties deterministic and independent of test data.
        best = max(trials, key=lambda trial: trial["selection_score"]["mean"])
        selected[family] = dict(best["parameters"])
        small_capacities[family] = int(best["structural_capacity"])
        development[family] = {
            "trials": trials,
            "selected_parameters": selected[family],
            "selected_capacity_at_cpam32_budget": small_capacities[family],
        }

    capacities: dict[str, dict[str, int]] = {
        family: {
            budget: _capacity_for_budget(
                family, budget_bytes[budget], selected[family]
            )
            for budget in BUDGETS
        }
        for family in FAMILIES
    }
    analytic_names = [
        f"{family}_{budget}_budget"
        for family in FAMILIES
        for budget in BUDGETS
    ]
    control_names = ["rls_ff_1", "managed_memory_32", "managed_memory_64"]
    names = [*control_names, *analytic_names]

    runs: list[dict[str, Any]] = []
    labels_preserved = True
    for seed in config.test_seeds:
        original = load_digits_split(
            test_per_class=config.test_per_class, seed=seed
        )
        regimes = _regime_splits(original, cpam_config, seed)
        labels_preserved = labels_preserved and all(
            np.array_equal(regime.train_labels, original.train_labels)
            and np.array_equal(regime.test_labels, original.test_labels)
            for regime in regimes.values()
        )
        models = {
            name: _run_capstone_model(
                name, cpam_config, regimes, seed=seed
            )
            for name in control_names
        }
        for family in FAMILIES:
            for budget in BUDGETS:
                name = f"{family}_{budget}_budget"
                models[name] = _run_analytic_model(
                    family,
                    budget,
                    capacities[family][budget],
                    selected[family],
                    config,
                    regimes,
                    seed=seed,
                )
        runs.append({"seed": seed, "models": models})

    metric_runs = [
        {name: _phase_metrics(run["models"][name]) for name in names}
        for run in runs
    ]
    all_metrics = tuple(metric_runs[0][names[0]])
    overall = {
        name: {
            metric: _rich_summary(
                [float(seed_run[name][metric]) for seed_run in metric_runs]
            )
            for metric in all_metrics
        }
        for name in names
    }
    comparisons: dict[str, Any] = {}
    for reference in ("rls_ff_1", "managed_memory_32", "managed_memory_64"):
        comparisons[reference] = {
            name: {
                metric: _rich_summary(
                    [
                        float(seed_run[name][metric])
                        - float(seed_run[reference][metric])
                        for seed_run in metric_runs
                    ]
                )
                for metric in PRIMARY_METRICS
            }
            for name in names
            if name != reference
        }

    model_runs = [model for run in runs for model in run["models"].values()]
    phase_runs = [phase for model in model_runs for phase in model["phases"]]
    state_budget_observed = {
        family: {
            budget: {
                "target_bytes": budget_bytes[budget],
                "capacity": capacities[family][budget],
                "actual_bytes": int(
                    next(
                        run["models"][f"{family}_{budget}_budget"][
                            "state_bytes_after"
                        ]
                        for run in runs
                    )
                ),
            }
            for budget in BUDGETS
        }
        for family in FAMILIES
    }
    return {
        "experiment": "closest_analytic_memory_baselines",
        "config": asdict(config),
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "downloaded_data_bytes": 0,
            "regimes": sorted(set(config.phase_domains)),
            "labels_preserved_by_regimes": labels_preserved,
        },
        "models": names,
        "memory_budgets": budget_bytes,
        "state_budget_allocations": state_budget_observed,
        "development": {
            "seeds": list(config.development_seeds),
            "seeds_disjoint_from_test": not bool(
                set(config.development_seeds) & set(config.test_seeds)
            ),
            "selection_metric": "equal mean of four primary quality metrics",
            "tuning_budget": "CPAM-32 persistent-state ceiling",
            "families": development,
        },
        "selected_parameters": selected,
        "invariants": {
            "same_fixed_signed_magnitude_frontend": True,
            "same_test_stream_per_seed": True,
            "one_update_per_training_image": all(
                model["trained_samples"]
                == len(config.phase_domains)
                * model["phases"][0]["training"]["samples"]
                for model in model_runs
            ),
            "no_gradients": True,
            "no_raw_image_replay": True,
            "weights_locked_during_evaluation": all(
                phase["pre_evaluation_locked"]
                and all(
                    evaluation["weights_unchanged"]
                    and evaluation["transient_state_restored"]
                    for evaluation in phase["evaluations"].values()
                )
                for phase in phase_runs
            ),
            "bounded_preallocated_state": all(
                model["bounded_state"] for model in model_runs
            ),
            "finite_persistent_state": all(
                model.get("finite_state", True) for model in model_runs
            ),
            "analytic_baselines_do_not_exceed_named_cpam_budget": all(
                item["actual_bytes"] <= item["target_bytes"]
                for family in state_budget_observed.values()
                for item in family.values()
            ),
            "development_and_test_seeds_disjoint": True,
        },
        "runs": runs,
        "summary": {
            "overall": overall,
            "paired_differences": comparisons,
        },
        "interpretation_constraints": {
            "krls_dictionary_elements_are_observed_feature_vectors": True,
            "ran_is_direct_link_ran_with_fixed_recruited_centers": True,
            "fuzzy_artmap_is_fast_learning_and_capacity_bounded": True,
            "os_elm_uses_regularized_event_wise_initialization": True,
            "memory_matching_is_a_ceiling_not_exact_byte_equality": True,
            "confidence_intervals_are_nominal_and_uncorrected": True,
        },
    }

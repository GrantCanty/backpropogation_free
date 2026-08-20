"""Factor-free cumulative representation-memory experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from no_backprop.digits import DigitsProtocol, load_digits_split
from no_backprop.experiment import (
    DigitsExperimentConfig,
    DigitsKind,
    run_digits_model,
)
from no_backprop.milestone6 import Milestone6Config, run_drift_model


@dataclass(frozen=True)
class FactorFreeMemoryConfig:
    """Configuration for the first cumulative fast/slow memory prototype."""

    hidden_size: int = 64
    test_per_class: int = 40
    seed: int = 29
    regularization: float = 1.0
    rank_bins: int = 16
    maturity_max_neurons: int = 32
    maturity_rbf_width: float = 0.05
    maturity_min_center_distance: float = 0.01
    key_prior_strength: float = 4.0
    key_minimum_variance: float = 4e-4
    key_maximum_variance: float = 3.6e-3
    protocols: tuple[DigitsProtocol, ...] = ("shuffled", "class_ordered")
    comparison_kinds: tuple[DigitsKind, ...] = (
        "cumulative_memory",
        "maturity",
        "maturity_entropy",
        "maturity_leverage",
        "maturity_probation",
        "probation_top4",
        "probation_top2",
        "probation_winner",
        "probation_top4_normalized",
        "probation_top2_normalized",
        "key_value",
        "key_value_entropy",
        "rls",
    )


def _digits_config(config: FactorFreeMemoryConfig) -> DigitsExperimentConfig:
    return DigitsExperimentConfig(
        hidden_size=config.hidden_size,
        test_per_class=config.test_per_class,
        seed=config.seed,
        # The RLS comparator is explicitly cumulative.  The new memory has no
        # forgetting-factor field at all.
        rls_forgetting_factor=1.0,
        cumulative_regularization=config.regularization,
        cumulative_rank_bins=config.rank_bins,
        maturity_max_neurons=config.maturity_max_neurons,
        maturity_rbf_width=config.maturity_rbf_width,
        maturity_min_center_distance=config.maturity_min_center_distance,
        key_prior_strength=config.key_prior_strength,
        key_minimum_variance=config.key_minimum_variance,
        key_maximum_variance=config.key_maximum_variance,
    )


def run_factor_free_quality(config: FactorFreeMemoryConfig) -> dict[str, Any]:
    digits_config = _digits_config(config)
    split = load_digits_split(
        test_per_class=config.test_per_class, seed=config.seed
    )
    return {
        protocol: {
            kind: run_digits_model(kind, protocol, digits_config, split=split)
            for kind in config.comparison_kinds
        }
        for protocol in config.protocols
    }


def run_factor_free_drift(config: FactorFreeMemoryConfig) -> dict[str, Any]:
    milestone_config = Milestone6Config(
        hidden_size=config.hidden_size,
        test_per_class=config.test_per_class,
        seed=config.seed,
        cumulative_regularization=config.regularization,
        cumulative_rank_bins=config.rank_bins,
        maturity_max_neurons=config.maturity_max_neurons,
        maturity_rbf_width=config.maturity_rbf_width,
        maturity_min_center_distance=config.maturity_min_center_distance,
        key_prior_strength=config.key_prior_strength,
        key_minimum_variance=config.key_minimum_variance,
        key_maximum_variance=config.key_maximum_variance,
    )
    return {
        "protocol": "original_to_inverted_to_original",
        "models": {
            "cumulative_memory": run_drift_model(
                "cumulative_memory", milestone_config
            ),
            "maturity": run_drift_model("maturity", milestone_config),
            "maturity_entropy": run_drift_model(
                "maturity_entropy", milestone_config
            ),
            "maturity_leverage": run_drift_model(
                "maturity_leverage", milestone_config
            ),
            "maturity_probation": run_drift_model(
                "maturity_probation", milestone_config
            ),
            "key_value": run_drift_model("key_value", milestone_config),
            "key_value_entropy": run_drift_model(
                "key_value_entropy", milestone_config
            ),
            "rls_no_discount": run_drift_model(
                "rls", milestone_config, rls_forgetting_factor=1.0
            ),
            "rls_discounted_0.999": run_drift_model(
                "rls", milestone_config, rls_forgetting_factor=0.999
            ),
        },
    }


def run_factor_free_memory(
    config: FactorFreeMemoryConfig = FactorFreeMemoryConfig(),
) -> dict[str, Any]:
    """Measure cumulative retention, fast adaptation, and compression."""

    quality = run_factor_free_quality(config)
    cumulative_runs = [
        models["cumulative_memory"]
        for models in quality.values()
        if "cumulative_memory" in models
    ]
    represented_samples = sum(
        run["memory_diagnostics"]["samples_in_slow_statistics"]
        for run in cumulative_runs
    )
    maturity_runs = [
        model
        for models in quality.values()
        for kind, model in models.items()
        if kind in (
            "maturity",
            "maturity_entropy",
            "maturity_leverage",
            "maturity_probation",
            "key_value",
            "key_value_entropy",
        )
    ]
    probation_runs = [
        model
        for models in quality.values()
        for kind, model in models.items()
        if kind
        in (
            "maturity_probation",
            "probation_top4",
            "probation_top2",
            "probation_winner",
            "probation_top4_normalized",
            "probation_top2_normalized",
        )
    ]
    return {
        "experiment": "factor_free_cumulative_representation_memory",
        "config": asdict(config),
        "invariants": {
            "forgetting_factor_in_new_model": None,
            "age_based_decay": False,
            "raw_samples_stored": 0,
            "every_observation_updates_slow_statistics": all(
                run["memory_diagnostics"]["samples_in_slow_statistics"]
                == run["trained_samples"]
                for run in cumulative_runs
            ),
            "represented_samples_across_protocol_runs": represented_samples,
            "single_prediction_path_in_maturity_models": True,
            "every_observation_updates_maturity_statistics": all(
                run["maturity_diagnostics"]["samples_in_cumulative_statistics"]
                == run["trained_samples"]
                for run in maturity_runs
            ),
            "every_observation_updates_leverage_statistics": all(
                run["maturity_diagnostics"]["samples_in_leverage_statistics"]
                == run["trained_samples"]
                for run in maturity_runs
            ),
            "maturity_models_store_raw_samples": any(
                run["maturity_diagnostics"]["stored_raw_samples"]
                for run in maturity_runs
            ),
            "probation_active_keys_are_frozen": all(
                run["maturity_diagnostics"]["active_keys_are_frozen"]
                for run in probation_runs
            ),
        },
        "quality": quality,
        "concept_drift": run_factor_free_drift(config),
    }

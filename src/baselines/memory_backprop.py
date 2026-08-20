"""Matched online-backpropagation controls for the recurring-memory capstone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from baselines.bptt import _module_bytes, _optimizer_bytes
from no_backprop.digits import build_digits_segments, load_digits_split
from no_backprop.memory_capstone import (
    MemoryCapstoneConfig,
    RegimeName,
    _phase_metrics,
    _regime_splits,
    _return_metrics,
    _run_model,
)
from no_backprop.spatial import PolarityConvolutionImageEncoder


BackpropArchitecture = Literal["linear", "mlp64"]
BACKPROP_ARCHITECTURES: tuple[BackpropArchitecture, ...] = ("linear", "mlp64")
NO_BACKPROP_MODELS = (
    "rls_ff_1",
    "managed_memory_32",
    "managed_memory_64",
)


@dataclass(frozen=True)
class MemoryBackpropComparisonConfig:
    """Development-tuned online Adam against fixed forward-only controls."""

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
    learning_rates: tuple[float, ...] = (
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
    )
    hidden_size: int = 64
    regularization: float = 1.0
    rbf_width: float = 0.05
    min_center_distance: float = 0.01
    candidate_capacity: int = 16
    translation_pixels: int = 1
    occlusion_size: int = 2

    def __post_init__(self) -> None:
        if not self.test_seeds or len(set(self.test_seeds)) != len(
            self.test_seeds
        ):
            raise ValueError("test_seeds must be non-empty and unique")
        if not self.development_seeds or len(set(self.development_seeds)) != len(
            self.development_seeds
        ):
            raise ValueError("development_seeds must be non-empty and unique")
        if set(self.test_seeds) & set(self.development_seeds):
            raise ValueError("development and test seeds must be disjoint")
        if (
            not self.learning_rates
            or len(set(self.learning_rates)) != len(self.learning_rates)
            or any(rate <= 0.0 for rate in self.learning_rates)
        ):
            raise ValueError("learning_rates must be positive and unique")
        if self.hidden_size != 64:
            raise ValueError("the state-near matched MLP requires hidden_size=64")
        _memory_config(self, self.test_seeds)


class OnlineFeatureModel(nn.Module):
    """Linear or one-hidden-layer classifier over fixed stable features."""

    def __init__(self, architecture: BackpropArchitecture, hidden_size: int) -> None:
        super().__init__()
        if architecture == "linear":
            self.network = nn.Linear(64, 10)
        elif architecture == "mlp64":
            self.network = nn.Sequential(
                nn.Linear(64, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 10),
            )
        else:  # pragma: no cover - Literal and config-generated callers
            raise ValueError(f"unknown architecture: {architecture}")

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError("features must have shape (batch, 64)")
        return self.network(features)


def _memory_config(
    config: MemoryBackpropComparisonConfig,
    seeds: tuple[int, ...],
) -> MemoryCapstoneConfig:
    return MemoryCapstoneConfig(
        seeds=seeds,
        test_per_class=config.test_per_class,
        phase_domains=config.phase_domains,
        mature_capacities=(32, 64),
        candidate_capacity=config.candidate_capacity,
        forgetting_factors=(1.0,),
        regularization=config.regularization,
        rbf_width=config.rbf_width,
        min_center_distance=config.min_center_distance,
        translation_pixels=config.translation_pixels,
        occlusion_size=config.occlusion_size,
    )


def _encode(
    encoder: PolarityConvolutionImageEncoder, images: np.ndarray
) -> Tensor:
    matrix = np.stack([encoder.encode(image) for image in images])
    return torch.as_tensor(matrix, dtype=torch.float64)


def _evaluate_locked(
    model: OnlineFeatureModel,
    encoder: PolarityConvolutionImageEncoder,
    images: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | bool]:
    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.eval()
    with torch.no_grad():
        logits = model(_encode(encoder, images))
        targets = torch.as_tensor(labels, dtype=torch.long)
        accuracy = float(torch.mean((torch.argmax(logits, 1) == targets).double()))
        cross_entropy = float(
            torch.nn.functional.cross_entropy(logits, targets)
        )
    unchanged = all(
        torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )
    if not unchanged:
        raise RuntimeError("online-backprop evaluation modified model weights")
    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "weights_unchanged": True,
        "transient_state_restored": True,
    }


def _empty_memory_diagnostics() -> dict[str, None | bool]:
    return {
        "mature_capacity": None,
        "active_neurons": None,
        "new_mature_neurons": None,
        "existing_neurons_reactivated": None,
        "existing_neuron_evidence_gain": None,
        "maximum_existing_center_shift": None,
        "capacity_full": False,
    }


def _run_backprop_model(
    architecture: BackpropArchitecture,
    learning_rate: float,
    config: MemoryBackpropComparisonConfig,
    regimes,
    *,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = OnlineFeatureModel(architecture, config.hidden_size).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    encoder = PolarityConvolutionImageEncoder(
        image_size=8, mode="signed_magnitude", seed=seed
    )
    parameter_pointers = {
        parameter.untyped_storage().data_ptr() for parameter in model.parameters()
    }
    peak_saved_activation_bytes = 0
    current_saved_pointers: set[int] = set()
    current_saved_bytes = 0

    def pack_hook(tensor: Tensor) -> Tensor:
        nonlocal current_saved_bytes, peak_saved_activation_bytes
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_pointers and pointer not in current_saved_pointers:
            current_saved_pointers.add(pointer)
            current_saved_bytes += storage.nbytes()
            peak_saved_activation_bytes = max(
                peak_saved_activation_bytes, current_saved_bytes
            )
        return tensor

    def unpack_hook(tensor: Tensor) -> Tensor:
        return tensor

    seen: set[RegimeName] = set()
    occurrences: dict[RegimeName, int] = {}
    phases: list[dict[str, Any]] = []
    trained_samples = 0
    training_seconds = 0.0
    for phase_index, domain in enumerate(config.phase_domains):
        split = regimes[domain]
        occurrence = occurrences.get(domain, 0) + 1
        occurrences[domain] = occurrence
        pre = _evaluate_locked(
            model, encoder, split.test_images, split.test_labels
        )
        segments = build_digits_segments(
            split.train_labels,
            protocol="shuffled",
            passes=1,
            seed=seed + 100 * (phase_index + 1),
        )
        correct = 0
        losses: list[float] = []
        phase_samples = 0
        model.train()
        started = perf_counter()
        for segment in segments:
            for index in segment.indices:
                label = int(split.train_labels[index])
                features = torch.as_tensor(
                    encoder.encode(split.train_images[index])[None, :],
                    dtype=torch.float64,
                )
                target = torch.tensor([label], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                current_saved_pointers.clear()
                current_saved_bytes = 0
                with torch.autograd.graph.saved_tensors_hooks(
                    pack_hook, unpack_hook
                ):
                    logits = model(features)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    correct += int(torch.argmax(logits, 1).item() == label)
                    loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                phase_samples += 1
        elapsed = perf_counter() - started
        training_seconds += elapsed
        trained_samples += phase_samples
        seen.add(domain)
        evaluations = {
            name: _evaluate_locked(
                model,
                encoder,
                regime.test_images,
                regime.test_labels,
            )
            for name, regime in regimes.items()
        }
        phases.append(
            {
                "phase_index": phase_index + 1,
                "domain": domain,
                "occurrence": occurrence,
                "pre_domain_accuracy": pre["accuracy"],
                "pre_evaluation_locked": pre["weights_unchanged"],
                "training": {
                    "samples": phase_samples,
                    "online_accuracy": correct / phase_samples,
                    "mean_cross_entropy": float(np.mean(losses)),
                },
                "training_seconds": elapsed,
                "post_domain_accuracy": evaluations[domain]["accuracy"],
                "mean_seen_domain_accuracy": float(
                    np.mean([evaluations[name]["accuracy"] for name in seen])
                ),
                "evaluations": evaluations,
                "memory": _empty_memory_diagnostics(),
            }
        )
    model_bytes = _module_bytes(model)
    optimizer_bytes = _optimizer_bytes(optimizer)
    persistent_state = model_bytes + optimizer_bytes
    final_evaluations = phases[-1]["evaluations"]
    return {
        "model": f"online_adam_{architecture}",
        "architecture": architecture,
        "learning_rate": learning_rate,
        "phases": phases,
        "trained_samples": trained_samples,
        "updates": trained_samples,
        "batch_size": 1,
        "replay_samples": 0,
        "uses_backpropagation": True,
        "model_bytes": model_bytes,
        "optimizer_bytes": optimizer_bytes,
        "peak_saved_activation_bytes": peak_saved_activation_bytes,
        "state_bytes_after": persistent_state,
        "peak_tracked_training_state_bytes": (
            persistent_state + peak_saved_activation_bytes
        ),
        "training_images_per_second": trained_samples / training_seconds,
        "final_mean_domain_accuracy": float(
            np.mean(
                [evaluation["accuracy"] for evaluation in final_evaluations.values()]
            )
        ),
        "capacity_filled_phase": None,
        "bounded_state": True,
    }


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": values,
    }


PRIMARY_METRICS = (
    "mean_online_accuracy",
    "first_shifted_online_accuracy",
    "mean_return_pre_accuracy",
    "mean_return_online_accuracy",
    "mean_return_post_accuracy",
    "mean_seen_domain_accuracy",
    "final_mean_domain_accuracy",
    "training_images_per_second",
    "state_bytes_after",
)


def _balanced_score(metrics: dict[str, float]) -> float:
    return float(
        np.mean(
            [
                metrics["first_shifted_online_accuracy"],
                metrics["mean_return_pre_accuracy"],
                metrics["mean_return_online_accuracy"],
                metrics["final_mean_domain_accuracy"],
            ]
        )
    )


def _tune_learning_rates(
    config: MemoryBackpropComparisonConfig,
) -> tuple[dict[BackpropArchitecture, float], dict[str, Any]]:
    memory_config = _memory_config(config, config.development_seeds)
    tuning: dict[str, Any] = {}
    selected: dict[BackpropArchitecture, float] = {}
    for architecture in BACKPROP_ARCHITECTURES:
        rates: dict[str, Any] = {}
        for rate in config.learning_rates:
            seed_metrics: list[dict[str, float]] = []
            for seed in config.development_seeds:
                original = load_digits_split(
                    test_per_class=config.test_per_class, seed=seed
                )
                regimes = _regime_splits(original, memory_config, seed)
                model = _run_backprop_model(
                    architecture, rate, config, regimes, seed=seed
                )
                metrics = _phase_metrics(model)
                seed_metrics.append(metrics)
            scores = [_balanced_score(metrics) for metrics in seed_metrics]
            rates[str(rate)] = {
                "balanced_continual_score": _summary(scores),
                "metrics": {
                    metric: _summary(
                        [values[metric] for values in seed_metrics]
                    )
                    for metric in PRIMARY_METRICS
                },
            }
        best = max(
            config.learning_rates,
            key=lambda rate: (
                rates[str(rate)]["balanced_continual_score"]["mean"],
                -rate,
            ),
        )
        selected[architecture] = best
        tuning[architecture] = {
            "selected_learning_rate": best,
            "selection_metric": (
                "mean(first_shifted_online, return_pre, return_online, final_mean)"
            ),
            "rates": rates,
        }
    return selected, tuning


def _run_test_seed(
    config: MemoryBackpropComparisonConfig,
    selected: dict[BackpropArchitecture, float],
    seed: int,
) -> dict[str, Any]:
    original = load_digits_split(test_per_class=config.test_per_class, seed=seed)
    memory_config = _memory_config(config, (seed,))
    regimes = _regime_splits(original, memory_config, seed)
    models = {
        name: _run_model(name, memory_config, regimes, seed=seed)
        for name in NO_BACKPROP_MODELS
    }
    for architecture in BACKPROP_ARCHITECTURES:
        name = f"online_adam_{architecture}"
        models[name] = _run_backprop_model(
            architecture,
            selected[architecture],
            config,
            regimes,
            seed=seed,
        )
    return {"seed": seed, "models": models}


def run_memory_backprop_comparison(
    config: MemoryBackpropComparisonConfig = MemoryBackpropComparisonConfig(),
) -> dict[str, Any]:
    """Tune on disjoint seeds, then compare online backprop and local memory."""

    selected, tuning = _tune_learning_rates(config)
    runs = [
        _run_test_seed(config, selected, seed) for seed in config.test_seeds
    ]
    names = (*NO_BACKPROP_MODELS, "online_adam_linear", "online_adam_mlp64")
    metric_runs = [
        {name: _phase_metrics(run["models"][name]) for name in names}
        for run in runs
    ]
    overall = {
        name: {
            metric: _summary(
                [values[name][metric] for values in metric_runs]
            )
            for metric in PRIMARY_METRICS
        }
        for name in names
    }
    paired: dict[str, Any] = {}
    for baseline in ("rls_ff_1", "managed_memory_32"):
        paired[baseline] = {
            name: {
                metric: _summary(
                    [
                        values[name][metric] - values[baseline][metric]
                        for values in metric_runs
                    ]
                )
                for metric in PRIMARY_METRICS
            }
            for name in names
            if name != baseline
        }
    returns = {
        domain: {
            name: {
                metric: _summary(
                    [
                        _return_metrics(run["models"][name], domain)[metric]
                        for run in runs
                    ]
                )
                for metric in _return_metrics(runs[0]["models"][name], domain)
            }
            for name in names
        }
        for domain in ("original", "inversion")
    }
    backprop_runs = [
        run["models"][name]
        for run in runs
        for name in ("online_adam_linear", "online_adam_mlp64")
    ]
    no_backprop_runs = [
        run["models"][name]
        for run in runs
        for name in NO_BACKPROP_MODELS
    ]
    return {
        "experiment": "matched_online_backprop_memory_comparison",
        "config": asdict(config),
        "development": {
            "seeds_disjoint_from_test": True,
            "selected_learning_rates": selected,
            "tuning": tuning,
        },
        "dataset": {
            "source": "sklearn.datasets.load_digits (bundled; no download)",
            "image_shape": [8, 8],
            "downloaded_data_bytes": 0,
            "phase_domains": list(config.phase_domains),
        },
        "models": list(names),
        "invariants": {
            "same_fixed_signed_magnitude_frontend": True,
            "same_test_stream_per_seed": True,
            "one_update_per_training_image": all(
                model["updates"] == model["trained_samples"]
                for model in backprop_runs
            )
            and all(
                model["trained_samples"]
                == sum(
                    phase["training"]["samples"] for phase in model["phases"]
                )
                for model in no_backprop_runs
            ),
            "batch_size_one_for_backprop": all(
                model["batch_size"] == 1 for model in backprop_runs
            ),
            "no_replay": all(
                model["replay_samples"] == 0 for model in backprop_runs
            ),
            "weights_locked_during_evaluation": all(
                phase["pre_evaluation_locked"]
                and all(
                    evaluation["weights_unchanged"]
                    for evaluation in phase["evaluations"].values()
                )
                for model in (*backprop_runs, *no_backprop_runs)
                for phase in model["phases"]
            ),
            "backprop_models_use_gradients": True,
            "local_models_use_gradients": False,
            "raw_samples_stored_by_models": 0,
        },
        "runs": runs,
        "summary": {
            "overall": overall,
            "paired_differences": paired,
            "returns": returns,
        },
    }

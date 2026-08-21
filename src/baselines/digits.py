"""Matched BPTT baseline for the bundled 8x8 handwritten digits."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from baselines.torch_resources import module_nbytes, optimizer_nbytes
from continual_core.datasets.digits import (
    DigitsProtocol,
    DigitsSplit,
    augment_digits_split,
    build_digits_segments,
    load_digits_split,
)


@dataclass(frozen=True)
class BPTTDigitsConfig:
    hidden_size: int = 64
    test_per_class: int = 40
    passes: int = 1
    augmentation_copies: int = 1
    augmentation_max_shift: int = 1
    augmentation_noise_std: float = 0.03
    seed: int = 29
    window: int = 100
    learning_rate: float = 0.001


class DigitsRNN(nn.Module):
    """Parameter-matched tanh RNN unrolled over the eight image rows."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Linear(8, hidden_size)
        self.recurrent_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, 10)
        nn.init.orthogonal_(self.recurrent_projection.weight, gain=0.88)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 3 or images.shape[1:] != (8, 8):
            raise ValueError("images must have shape (batch, 8, 8)")
        state = torch.zeros(
            (images.shape[0], self.hidden_size),
            dtype=images.dtype,
            device=images.device,
        )
        for row_index in range(8):
            state = torch.tanh(
                self.input_projection(images[:, row_index, :])
                + self.recurrent_projection(state)
            )
        return self.readout(state)


def _evaluate_locked(
    model: DigitsRNN, images: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Run inference with autograd disabled and prove parameters do not change."""

    before = [parameter.detach().clone() for parameter in model.parameters()]
    model.eval()
    with torch.no_grad():
        inputs = torch.as_tensor(images, dtype=torch.float32)
        targets = torch.as_tensor(labels, dtype=torch.long)
        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1)
        correct = predictions.eq(targets).cpu().numpy().astype(np.float64)
        losses = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"
        ).cpu().numpy()
    unchanged = all(
        torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )
    if not unchanged:
        raise RuntimeError("BPTT evaluation modified model weights")
    per_class = {
        str(class_index): float(np.mean(correct[labels == class_index]))
        for class_index in np.unique(labels)
    }
    return {
        "accuracy": float(np.mean(correct)),
        "cross_entropy": float(np.mean(losses)),
        "worst_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
        "weights_unchanged": True,
    }


def _forgetting(history: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoints = history[1:]
    final = history[-1]["per_class_accuracy"]
    per_class: dict[str, float] = {}
    for class_key in final:
        class_index = int(class_key)
        eligible = [
            item["per_class_accuracy"][class_key]
            for item in checkpoints
            if class_index in item["classes_seen"]
        ]
        peak = max(eligible) if eligible else final[class_key]
        per_class[class_key] = float(max(0.0, peak - final[class_key]))
    return {
        "mean": float(np.mean(list(per_class.values()))),
        "maximum": float(max(per_class.values())),
        "per_class": per_class,
    }


def run_bptt_digits(
    config: BPTTDigitsConfig,
    *,
    protocol: DigitsProtocol = "shuffled",
    split: DigitsSplit | None = None,
) -> dict[str, Any]:
    """Train with one Adam/BPTT update per image on a shuffled stream."""

    if protocol not in (
        "shuffled",
        "shuffled_repeated",
        "shuffled_augmented",
    ):
        raise ValueError("the traditional baseline only supports shuffled protocols")
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    data = split or load_digits_split(
        test_per_class=config.test_per_class, seed=config.seed
    )
    segments = build_digits_segments(
        data.train_labels,
        protocol=protocol,
        passes=config.passes,
        seed=config.seed + 2,
    )
    model = DigitsRNN(config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    training_correct: list[float] = []
    training_losses: list[float] = []
    segment_results: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    classes_seen: set[int] = set()
    trained_samples = 0
    training_seconds = 0.0
    evaluation_seconds = 0.0

    started = time.perf_counter()
    initial_evaluation = _evaluate_locked(model, data.test_images, data.test_labels)
    evaluation_seconds += time.perf_counter() - started
    evaluation_history.append(
        {
            "trained_samples": 0,
            "pass": -1,
            "segment": -1,
            "classes_seen": [],
            **initial_evaluation,
        }
    )

    for segment in segments:
        segment_correct: list[float] = []
        segment_losses: list[float] = []
        model.train()
        started = time.perf_counter()
        for index in segment.indices:
            label = int(data.train_labels[index])
            image = torch.as_tensor(
                data.train_images[index][None, :, :], dtype=torch.float32
            )
            target = torch.tensor([label], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, target)
            is_correct = float(int(torch.argmax(logits, dim=1).item()) == label)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
            segment_correct.append(is_correct)
            segment_losses.append(loss_value)
            training_correct.append(is_correct)
            training_losses.append(loss_value)
            classes_seen.add(label)
            trained_samples += 1
        training_seconds += time.perf_counter() - started

        width = min(config.window, len(segment_correct))
        segment_results.append(
            {
                "pass": segment.pass_index,
                "segment": segment.segment_index,
                "samples": len(segment.indices),
                "accuracy": float(np.mean(segment_correct)),
                "head_accuracy": float(np.mean(segment_correct[:width])),
                "tail_accuracy": float(np.mean(segment_correct[-width:])),
                "cross_entropy": float(np.mean(segment_losses)),
            }
        )
        started = time.perf_counter()
        evaluation = _evaluate_locked(model, data.test_images, data.test_labels)
        evaluation_seconds += time.perf_counter() - started
        evaluation_history.append(
            {
                "trained_samples": trained_samples,
                "pass": segment.pass_index,
                "segment": segment.segment_index,
                "classes_seen": sorted(classes_seen),
                **evaluation,
            }
        )

    final_evaluation = evaluation_history[-1]
    width = min(config.window, len(training_correct))
    return {
        "model": "bptt_adam",
        "protocol": protocol,
        "config": asdict(config),
        "trained_samples": trained_samples,
        "updates": trained_samples,
        "batch_size": 1,
        "unroll_steps": 8,
        "online_accuracy": float(np.mean(training_correct)),
        "tail_online_accuracy": float(np.mean(training_correct[-width:])),
        "online_cross_entropy": float(np.mean(training_losses)),
        "initial_test_accuracy": initial_evaluation["accuracy"],
        "final_test_accuracy": final_evaluation["accuracy"],
        "final_test_cross_entropy": final_evaluation["cross_entropy"],
        "final_worst_class_accuracy": final_evaluation["worst_class_accuracy"],
        "forgetting": _forgetting(evaluation_history),
        "segments": segment_results,
        "evaluation_history": evaluation_history,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "training_images_per_second": trained_samples / training_seconds,
        "model_bytes": module_nbytes(model),
        "optimizer_bytes": optimizer_nbytes(optimizer),
        "weights_locked_during_evaluation": all(
            item["weights_unchanged"] for item in evaluation_history
        ),
    }


def run_bptt_digits_experiment(config: BPTTDigitsConfig) -> dict[str, Any]:
    """Run the traditional baseline on matched plain and augmented streams."""

    split = load_digits_split(test_per_class=config.test_per_class, seed=config.seed)
    augmented = augment_digits_split(
        split,
        copies=config.augmentation_copies,
        max_shift=config.augmentation_max_shift,
        noise_std=config.augmentation_noise_std,
        seed=config.seed + 3,
    )
    return {
        "experiment": "bptt_digits_8x8",
        "config": asdict(config),
        "protocols": {
            "shuffled": run_bptt_digits(
                config, protocol="shuffled", split=split
            ),
            "shuffled_augmented": run_bptt_digits(
                config, protocol="shuffled_augmented", split=augmented
            ),
        },
    }

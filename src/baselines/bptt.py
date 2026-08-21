"""Matched recurrent baseline trained with truncated backpropagation through time."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from baselines.torch_resources import module_nbytes, optimizer_nbytes
from continual_core.streams import iter_nonstationary_signal


@dataclass(frozen=True)
class BPTTConfig:
    steps: int = 3_000
    regime_length: int = 750
    hidden_size: int = 64
    window: int = 32
    seed: int = 7
    learning_rate: float = 0.01


class TinyRNN(nn.Module):
    """A dense tanh RNN with dimensions matched to the online reservoir."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.recurrent_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, output_size)
        nn.init.orthogonal_(self.recurrent_projection.weight, gain=0.9)

    def step(self, observation: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        state = torch.tanh(
            self.input_projection(observation) + self.recurrent_projection(state)
        )
        return self.readout(state), state


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return None


def run_bptt_signal(config: BPTTConfig) -> dict[str, Any]:
    """Train prequentially, updating after each retained BPTT window."""

    if config.window <= 0:
        raise ValueError("window must be positive")
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    model = TinyRNN(1, config.hidden_size, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    state = torch.zeros(config.hidden_size)
    losses: list[Tensor] = []
    squared_errors: list[float] = []
    update_count = 0
    current_saved_bytes = 0
    peak_saved_bytes = 0
    saved_storage_pointers: set[int] = set()
    parameter_storage_pointers = {
        parameter.untyped_storage().data_ptr() for parameter in model.parameters()
    }

    def pack_hook(tensor: Tensor) -> Tensor:
        nonlocal current_saved_bytes, peak_saved_bytes
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer in parameter_storage_pointers or pointer in saved_storage_pointers:
            return tensor
        saved_storage_pointers.add(pointer)
        current_saved_bytes += storage.nbytes()
        peak_saved_bytes = max(peak_saved_bytes, current_saved_bytes)
        return tensor

    def unpack_hook(tensor: Tensor) -> Tensor:
        return tensor

    rss_before = _process_rss_bytes()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        for event in iter_nonstationary_signal(
            config.steps,
            regime_length=config.regime_length,
            seed=config.seed,
        ):
            observation = torch.from_numpy(event.observation).to(dtype=torch.float32)
            target = torch.from_numpy(event.target).to(dtype=torch.float32)
            prediction, state = model.step(observation, state)
            error = target - prediction
            squared_errors.append(float(torch.mean(error.square()).detach()))
            losses.append(torch.mean(error.square()))
            if len(losses) == config.window:
                torch.stack(losses).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state = state.detach()
                losses.clear()
                update_count += 1
                current_saved_bytes = 0
                saved_storage_pointers.clear()
        if losses:
            torch.stack(losses).mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_count += 1
            current_saved_bytes = 0
            saved_storage_pointers.clear()
    elapsed = time.perf_counter() - started
    rss_after = _process_rss_bytes()
    errors = np.asarray(squared_errors, dtype=np.float64)
    metric_window = min(100, len(errors))
    model_bytes = module_nbytes(model)
    optimizer_bytes = optimizer_nbytes(optimizer)
    return {
        "model": "bptt",
        "config": asdict(config),
        "mse": float(np.mean(errors)),
        "head_mse": float(np.mean(errors[:metric_window])),
        "tail_mse": float(np.mean(errors[-metric_window:])),
        "elapsed_seconds": elapsed,
        "steps_per_second": config.steps / elapsed,
        "updates": update_count,
        "model_bytes": model_bytes,
        "optimizer_bytes": optimizer_bytes,
        "peak_saved_activation_bytes": peak_saved_bytes,
        "training_state_bytes": model_bytes + optimizer_bytes + peak_saved_bytes,
        "rss_delta_bytes": (
            None if rss_before is None or rss_after is None else rss_after - rss_before
        ),
    }

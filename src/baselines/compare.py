"""Systems comparison between online local learning and matched BPTT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from baselines.bptt import BPTTConfig, run_bptt_signal
from no_backprop.experiment import (
    SignalExperimentConfig,
    run_signal_model,
)


@dataclass(frozen=True)
class SystemsComparisonConfig:
    steps: int = 3_000
    regime_length: int = 750
    hidden_size: int = 64
    seed: int = 7
    bptt_windows: tuple[int, ...] = (8, 32, 128)


def run_systems_comparison(config: SystemsComparisonConfig) -> dict[str, Any]:
    signal_config = SignalExperimentConfig(
        steps=config.steps,
        regime_length=config.regime_length,
        hidden_size=config.hidden_size,
        seed=config.seed,
    )
    online = {
        kind: run_signal_model(kind, signal_config) for kind in ("lms", "rls")
    }
    bptt = {
        str(window): run_bptt_signal(
            BPTTConfig(
                steps=config.steps,
                regime_length=config.regime_length,
                hidden_size=config.hidden_size,
                window=window,
                seed=config.seed,
            )
        )
        for window in config.bptt_windows
    }
    return {
        "experiment": "systems_comparison",
        "config": asdict(config),
        "online": online,
        "bptt": bptt,
    }

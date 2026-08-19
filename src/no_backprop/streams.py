"""Deterministic synthetic streams for online-learning experiments."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from no_backprop.protocol import StreamEvent


@dataclass(frozen=True)
class SignalRegime:
    """Parameters for one segment of a nonstationary oscillator."""

    angular_frequency: float
    amplitude: float = 1.0
    noise_std: float = 0.02


DEFAULT_SIGNAL_REGIMES: tuple[SignalRegime, ...] = (
    SignalRegime(angular_frequency=0.075, amplitude=1.0),
    SignalRegime(angular_frequency=0.19, amplitude=0.7),
    SignalRegime(angular_frequency=0.035, amplitude=1.25),
    SignalRegime(angular_frequency=0.075, amplitude=1.0),
)


def iter_nonstationary_signal(
    steps: int,
    *,
    regime_length: int = 750,
    regimes: Sequence[SignalRegime] = DEFAULT_SIGNAL_REGIMES,
    seed: int = 0,
) -> Iterator[StreamEvent]:
    """Yield a continuous noisy signal whose dynamics change without warning.

    Only the current and next samples are retained, so generator memory is
    independent of ``steps``.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if regime_length <= 1:
        raise ValueError("regime_length must be greater than one")
    if not regimes:
        raise ValueError("at least one regime is required")

    rng = np.random.default_rng(seed)
    phase = 0.0

    def sample(step: int) -> tuple[float, int]:
        nonlocal phase
        regime_index = (step // regime_length) % len(regimes)
        regime = regimes[regime_index]
        phase += regime.angular_frequency
        value = regime.amplitude * np.sin(phase)
        value += float(rng.normal(0.0, regime.noise_std))
        return float(value), regime_index

    current, current_regime = sample(0)
    for step in range(steps):
        following, following_regime = sample(step + 1)
        yield StreamEvent(
            step=step,
            observation=np.array([current], dtype=np.float64),
            target=np.array([following], dtype=np.float64),
            regime=current_regime,
            change_point=step > 0 and current_regime != previous_regime,
        )
        previous_regime = current_regime
        current = following
        current_regime = following_regime


@dataclass(frozen=True)
class DelayedAssociationConfig:
    episodes: int = 500
    delay: int = 8
    distractor_std: float = 0.15
    seed: int = 0


def iter_delayed_association(
    config: DelayedAssociationConfig,
) -> Iterator[StreamEvent]:
    """Yield cue/distractor/query events with feedback only at query time.

    Observation channels are ``[cue_value, distractor, is_query]``. Targets are
    NaN before the query, which tells the learner to evolve state without a
    supervised update.
    """

    if config.episodes <= 0 or config.delay < 1:
        raise ValueError("episodes and delay must be positive")
    rng = np.random.default_rng(config.seed)
    step = 0
    for episode in range(config.episodes):
        cue = float(rng.choice((-1.0, 1.0)))
        yield StreamEvent(
            step=step,
            observation=np.array([cue, 0.0, 0.0], dtype=np.float64),
            target=np.array([np.nan], dtype=np.float64),
            regime=episode,
        )
        step += 1
        for _ in range(config.delay):
            distractor = float(rng.normal(0.0, config.distractor_std))
            yield StreamEvent(
                step=step,
                observation=np.array([0.0, distractor, 0.0], dtype=np.float64),
                target=np.array([np.nan], dtype=np.float64),
                regime=episode,
            )
            step += 1
        yield StreamEvent(
            step=step,
            observation=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            target=np.array([cue], dtype=np.float64),
            regime=episode,
        )
        step += 1

import pytest

pytest.importorskip("torch")

from baselines.bptt import BPTTConfig, run_bptt_signal


def test_bptt_activation_memory_grows_with_window() -> None:
    short = run_bptt_signal(
        BPTTConfig(steps=96, regime_length=48, hidden_size=8, window=4, seed=2)
    )
    long = run_bptt_signal(
        BPTTConfig(steps=96, regime_length=48, hidden_size=8, window=16, seed=2)
    )
    assert long["peak_saved_activation_bytes"] > short["peak_saved_activation_bytes"]
    assert short["updates"] > long["updates"]
    assert short["training_state_bytes"] > 0

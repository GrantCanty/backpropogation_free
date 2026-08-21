import pytest

pytest.importorskip("torch")

from baselines.bptt import BPTTConfig, run_bptt_signal
from baselines.digits import BPTTDigitsConfig, DigitsRNN, run_bptt_digits
from no_backprop.digits import DigitsSplit
from no_backprop.experiment import DigitsExperimentConfig, build_digits_learner


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


def test_bptt_digits_uses_locked_evaluation_and_matched_online_updates() -> None:
    import numpy as np

    rng = np.random.default_rng(4)
    train_labels = np.repeat(np.arange(2), 5)
    test_labels = np.repeat(np.arange(2), 3)
    split = DigitsSplit(
        train_images=rng.uniform(size=(len(train_labels), 8, 8)),
        train_labels=train_labels,
        test_images=rng.uniform(size=(len(test_labels), 8, 8)),
        test_labels=test_labels,
    )
    result = run_bptt_digits(
        BPTTDigitsConfig(hidden_size=8, window=3, seed=4),
        protocol="shuffled",
        split=split,
    )
    assert result["updates"] == len(train_labels)
    assert result["batch_size"] == 1
    assert result["unroll_steps"] == 8
    assert result["weights_locked_during_evaluation"]


def test_digits_rnns_have_matched_parameter_counts() -> None:
    bptt = DigitsRNN(hidden_size=8)
    local = build_digits_learner("lms", DigitsExperimentConfig(hidden_size=8))
    bptt_count = sum(parameter.numel() for parameter in bptt.parameters())
    local_count = (
        local.input_weights.size
        + local.recurrent_weights.size
        + local.bias.size
        + local.readout.weights.size
    )
    assert bptt_count == local_count

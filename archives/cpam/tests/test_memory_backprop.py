import numpy as np
import pytest

pytest.importorskip("torch")

import baselines.memory_backprop as memory_backprop
from baselines.memory_backprop import (
    MemoryBackpropComparisonConfig,
    run_memory_backprop_comparison,
)
from no_backprop.digits import DigitsSplit


def _small_split(seed: int = 3) -> DigitsSplit:
    rng = np.random.default_rng(seed)
    train_labels = np.repeat(np.arange(2), 6)
    test_labels = np.repeat(np.arange(2), 2)
    return DigitsSplit(
        train_images=rng.uniform(size=(len(train_labels), 8, 8)),
        train_labels=train_labels,
        test_images=rng.uniform(size=(len(test_labels), 8, 8)),
        test_labels=test_labels,
    )


def test_online_backprop_comparison_is_matched_and_tuned_off_test(
    monkeypatch,
) -> None:
    split = _small_split(7)
    monkeypatch.setattr(memory_backprop, "load_digits_split", lambda **_: split)
    result = run_memory_backprop_comparison(
        MemoryBackpropComparisonConfig(
            test_seeds=(7,),
            development_seeds=(2,),
            test_per_class=1,
            phase_domains=(
                "original",
                "inversion",
                "original",
                "inversion",
            ),
            learning_rates=(0.0003, 0.001),
        )
    )

    assert result["development"]["seeds_disjoint_from_test"]
    assert set(result["development"]["selected_learning_rates"]) == {
        "linear",
        "mlp64",
    }
    assert set(result["models"]) == {
        "rls_ff_1",
        "managed_memory_32",
        "managed_memory_64",
        "online_adam_linear",
        "online_adam_mlp64",
    }
    invariants = result["invariants"]
    assert invariants["same_fixed_signed_magnitude_frontend"]
    assert invariants["same_test_stream_per_seed"]
    assert invariants["one_update_per_training_image"]
    assert invariants["batch_size_one_for_backprop"]
    assert invariants["no_replay"]
    assert invariants["weights_locked_during_evaluation"]
    assert invariants["backprop_models_use_gradients"]
    assert not invariants["local_models_use_gradients"]
    assert invariants["raw_samples_stored_by_models"] == 0
    mlp = result["runs"][0]["models"]["online_adam_mlp64"]
    memory = result["runs"][0]["models"]["managed_memory_32"]
    assert mlp["updates"] == mlp["trained_samples"]
    assert mlp["peak_saved_activation_bytes"] > 0
    assert mlp["optimizer_bytes"] > 0
    assert (
        abs(mlp["state_bytes_after"] - memory["state_bytes_after"])
        / memory["state_bytes_after"]
        < 0.10
    )


def test_backprop_comparison_rejects_confounded_configuration() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        MemoryBackpropComparisonConfig(
            test_seeds=(3,), development_seeds=(3,)
        )
    with pytest.raises(ValueError, match="positive and unique"):
        MemoryBackpropComparisonConfig(
            test_seeds=(3,),
            development_seeds=(2,),
            learning_rates=(0.001, 0.001),
        )
    with pytest.raises(ValueError, match="hidden_size=64"):
        MemoryBackpropComparisonConfig(
            test_seeds=(3,), development_seeds=(2,), hidden_size=32
        )

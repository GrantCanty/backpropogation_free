import numpy as np

from no_backprop.experiment import ContinualExperimentConfig, run_continual_experiment
from no_backprop.readouts import FastSlowLMSReadout
from no_backprop.streams import (
    ContinualClassificationConfig,
    iter_continual_classification,
)


def test_continual_stream_repeats_first_context() -> None:
    config = ContinualClassificationConfig(steps=40, context_length=10, seed=2)
    events = list(iter_continual_classification(config))
    assert [events[index].regime for index in (0, 10, 20, 30)] == [0, 1, 2, 0]
    assert [events[index].change_point for index in (0, 10, 20, 30)] == [
        False,
        True,
        True,
        True,
    ]


def test_fast_slow_readout_has_two_bounded_timescales() -> None:
    readout = FastSlowLMSReadout(3, 2, learning_rate=0.2)
    features = np.array([1.0, -0.5, 1.0])
    target = np.array([1.0, 0.0])
    readout.update(features, target, readout.predict(features))
    assert np.linalg.norm(readout.fast_weights) > 0.0
    assert np.linalg.norm(readout.slow_weights) > 0.0
    assert readout.state_nbytes == readout.fast_weights.nbytes + readout.slow_weights.nbytes


def test_continual_ablation_models_are_bounded() -> None:
    result = run_continual_experiment(
        ContinualExperimentConfig(
            steps=320,
            context_length=80,
            hidden_size=12,
            input_size=6,
            seed=3,
            window=20,
        )
    )
    assert set(result["models"]) == {
        "fixed",
        "eligibility",
        "gated",
        "fast_slow",
    }
    assert all(model["bounded_state"] for model in result["models"].values())

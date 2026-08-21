import numpy as np

from baselines.lms import LMSReadout
from baselines.rls import RLSReadout
from continual_core.evaluation import DirectUpdateAdapter
from experiments.online_classification import MethodSetup, run_comparison


def target(label: int) -> np.ndarray:
    value = np.zeros(2, dtype=np.float64)
    value[label] = 1.0
    return value


def test_factory_driven_comparison_uses_matched_events_and_versioned_results() -> None:
    observations = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    events = tuple(zip(observations, (target(0), target(1))))
    adapter = DirectUpdateAdapter()
    result = run_comparison(
        experiment="fixture",
        methods={
            "lms": MethodSetup(
                lambda: LMSReadout(2, 2, learning_rate=1.0),
                adapter,
                {"gradients": False},
            ),
            "rls": MethodSetup(
                lambda: RLSReadout(2, 2, forgetting_factor=1.0),
                adapter,
                {"gradients": False, "forgetting_factor": 1.0},
            ),
        },
        training_events=events,
        evaluation_observations=observations,
        evaluation_labels=(0, 1),
        encode_target=target,
        seed=7,
    )

    assert result["schema_version"] == 1
    assert set(result["methods"]) == {"lms", "rls"}
    for method in result["methods"].values():
        assert method["training"]["samples"] == 2
        assert method["evaluation"]["weights_unchanged"]

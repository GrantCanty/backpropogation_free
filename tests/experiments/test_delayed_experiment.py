from experiments.streams import DelayedExperimentConfig, run_delayed_experiment


def test_delayed_experiment_is_online_and_bounded() -> None:
    result = run_delayed_experiment(
        DelayedExperimentConfig(episodes=120, delay=3, hidden_size=12, seed=5)
    )
    assert set(result["models"]) == {"fixed", "eligibility"}
    assert all(model["bounded_state"] for model in result["models"].values())
    assert result["models"]["eligibility"]["diagnostics"]["updates"] == 120

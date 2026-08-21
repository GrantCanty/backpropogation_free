from experiments.streams import SignalExperimentConfig, run_signal_experiment


def test_online_models_learn_nonstationary_signal() -> None:
    result = run_signal_experiment(
        SignalExperimentConfig(
            steps=800,
            regime_length=200,
            hidden_size=20,
            seed=11,
            window=50,
        )
    )
    models = result["models"]
    assert models["lms"]["mse"] < models["frozen"]["mse"]
    assert models["rls"]["mse"] < models["frozen"]["mse"]
    assert all(model["bounded_state"] for model in models.values())

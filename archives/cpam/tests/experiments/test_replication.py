from experiments.legacy import ContinualExperimentConfig, DelayedExperimentConfig
from experiments.replication import ReplicationConfig, run_replication


def test_replication_aggregates_seeded_runs() -> None:
    result = run_replication(
        ReplicationConfig(
            seeds=(2, 5),
            delayed=DelayedExperimentConfig(
                episodes=40, delay=2, hidden_size=8, window=10
            ),
            continual=ContinualExperimentConfig(
                steps=160,
                context_length=40,
                hidden_size=8,
                input_size=5,
                window=10,
            ),
        )
    )
    assert result["config"]["seeds"] == [2, 5]
    values = result["delayed"]["eligibility"]["mse"]["values"]
    assert len(values) == 2
    assert "fast_slow_retention_improvement" in result["continual"]

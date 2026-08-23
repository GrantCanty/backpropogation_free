from continual_core.metrics import (
    PrequentialMetrics,
    classification_plasticity_summary,
    sample_efficiency_steps,
)


def test_rolling_metric_is_bounded_in_output_resolution() -> None:
    metrics = PrequentialMetrics()
    for step in range(20):
        metrics.squared_errors.append(float(step))
    curve = metrics.rolling_mse(window=5, stride=4)
    assert curve[-1]["step"] == 20
    assert curve[-1]["mse"] == 17.0


def test_classification_plasticity_summary_tracks_acquisition_and_forgetting() -> None:
    def evaluation(all_accuracy, class_0, class_1):
        return {
            "all": {"accuracy": all_accuracy, "worst_class_accuracy": min(class_0, class_1)},
            "class_0": {"accuracy": class_0},
            "class_1": {"accuracy": class_1},
        }

    training = {
        "samples": 100,
        "sample_efficiency": [
            {"samples": 10, "cumulative_online_accuracy": 0.4},
            {"samples": 100, "cumulative_online_accuracy": 0.8},
        ],
        "segments": [
            {"focus_class": 0, "adaptation_delta": 0.2, "events_to_90pct_tail_accuracy": 20},
            {"focus_class": 1, "adaptation_delta": 0.3, "events_to_90pct_tail_accuracy": 30},
            {"focus_class": 0, "adaptation_delta": 0.1, "events_to_90pct_tail_accuracy": 10},
        ],
        "checkpoints": [
            {"evaluation_sets": evaluation(0.5, 0.9, 0.1)},
            {"evaluation_sets": evaluation(0.7, 0.7, 0.8)},
            {"evaluation_sets": evaluation(0.8, 0.85, 0.75)},
        ],
    }

    result = classification_plasticity_summary(training)
    assert sample_efficiency_steps(100) == (1, 5, 10, 25, 50, 100)
    assert result["focused_segment_count"] == 3
    assert result["recurring_visit_count"] == 1
    assert result["average_forgetting"] > 0.0
    assert result["mean_relearning_gain"] > 0.0
    assert result["final_locked_accuracy"] == 0.8

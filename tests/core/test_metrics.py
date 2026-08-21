from continual_core.metrics import PrequentialMetrics


def test_rolling_metric_is_bounded_in_output_resolution() -> None:
    metrics = PrequentialMetrics()
    for step in range(20):
        metrics.squared_errors.append(float(step))
    curve = metrics.rolling_mse(window=5, stride=4)
    assert curve[-1]["step"] == 20
    assert curve[-1]["mse"] == 17.0

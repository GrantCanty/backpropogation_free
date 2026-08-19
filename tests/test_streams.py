import numpy as np

from no_backprop.streams import iter_nonstationary_signal


def test_signal_stream_is_deterministic_and_marks_changes() -> None:
    left = list(iter_nonstationary_signal(12, regime_length=4, seed=3))
    right = list(iter_nonstationary_signal(12, regime_length=4, seed=3))
    assert [event.change_point for event in left] == [False] * 4 + [True] + [False] * 3 + [True] + [False] * 3
    for first, second in zip(left, right):
        np.testing.assert_array_equal(first.observation, second.observation)
        np.testing.assert_array_equal(first.target, second.target)

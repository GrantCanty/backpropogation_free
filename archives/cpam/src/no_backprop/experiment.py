"""Compatibility exports for historical experiment runners."""

from continual_core.results import write_json_result
from experiments.legacy import *  # noqa: F403
from experiments.legacy import (
    _digit_target,
    _evaluate_digits,
    _evaluate_digits_locked,
    _learner_training_arrays,
    _process_digit_image,
)

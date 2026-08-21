import json
from pathlib import Path

import pytest

from continual_core import __version__
from experiments.__main__ import build_config, main, read_config
from experiments.streams import (
    ContinualExperimentConfig,
    DelayedExperimentConfig,
    SignalExperimentConfig,
)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_starts() -> None:
    assert main([]) == 0


def test_self_identifying_config_loads_all_fields_and_accepts_overrides(
    tmp_path,
) -> None:
    path = tmp_path / "signal.json"
    path.write_text(
        json.dumps(
            {
                "benchmark": "signal",
                "steps": 12,
                "hidden_size": 5,
                "rls_forgetting_factor": 1.0,
            }
        )
    )
    benchmark, values = read_config(path)
    config = build_config(benchmark, values, {"steps": 8})
    assert config == SignalExperimentConfig(
        steps=8,
        hidden_size=5,
        rls_forgetting_factor=1.0,
    )


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown delayed configuration"):
        build_config("delayed", {"not_a_setting": 3})


def test_each_checked_in_config_is_executable() -> None:
    expected = {
        "signal_mvp.json": SignalExperimentConfig,
        "delayed_mvp.json": DelayedExperimentConfig,
        "continual_mvp.json": ContinualExperimentConfig,
    }
    config_root = Path(__file__).parents[2] / "configs"
    for filename, config_type in expected.items():
        benchmark, values = read_config(config_root / filename)
        assert isinstance(build_config(benchmark, values), config_type)

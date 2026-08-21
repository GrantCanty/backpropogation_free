import json

from continual_core.results import RESULT_SCHEMA_VERSION, result_envelope, write_json_result


def test_result_envelope_and_atomic_writer(tmp_path) -> None:
    result = result_envelope(
        experiment="fixture",
        method="rls",
        seed=3,
        assumptions={"replay": False},
        payload={"quality_metrics": {"accuracy": 0.5}},
    )
    destination = write_json_result(result, tmp_path / "result.json")
    loaded = json.loads(destination.read_text())
    assert loaded["schema_version"] == RESULT_SCHEMA_VERSION
    assert loaded["method_assumptions"] == {"replay": False}


def test_writer_versions_legacy_results(tmp_path) -> None:
    destination = write_json_result({"experiment": "old"}, tmp_path / "old.json")
    loaded = json.loads(destination.read_text())
    assert loaded == {
        "experiment": "old",
        "schema_version": RESULT_SCHEMA_VERSION,
    }

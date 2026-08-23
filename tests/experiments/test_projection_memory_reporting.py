import json
import pytest

from experiments.projection_memory_reporting import (
    summarize_projection_results,
    write_projection_summary,
)


def _run(
    condition: str,
    final_accuracy: float,
    *,
    representation_bytes: int = 100,
    solver_bytes: int = 200,
    total_bytes: int = 300,
    event_microseconds: float = 20.0,
) -> dict:
    evaluation = {
        "all": {"accuracy": final_accuracy, "worst_class_accuracy": final_accuracy - 0.1},
        "class_0": {"accuracy": final_accuracy},
    }
    return {
        "completed": True,
        "condition": condition,
        "configuration_hash": f"hash-{condition}",
        "matched_problem_sha256": "problem-3",
        "seed": 3,
        "parameters": {"feature": "sparse" if condition.startswith("sparse") else "dense"},
        "runtime_environment": {"timing_session_id": "fixture"},
        "resources": {
            "measurement_schema_version": 2,
            "representation_bytes": representation_bytes,
            "solver_bytes": solver_bytes,
            "total_persistent_bytes": total_bytes,
            "throughput_events_per_second": 50.0,
            "solver_update_latency": {"median_microseconds": 10.0},
            "end_to_end_event_latency": {"median_microseconds": event_microseconds},
        },
        "training": {
            "samples": 10,
            "online_accuracy": final_accuracy - 0.1,
            "nonfinite_state_values": 0,
            "sample_efficiency": [
                {"samples": 1, "cumulative_online_accuracy": 0.5},
                {"samples": 10, "cumulative_online_accuracy": final_accuracy - 0.1},
            ],
            "segments": [
                {"focus_class": 0, "adaptation_delta": 0.2, "events_to_90pct_tail_accuracy": 2}
            ],
            "checkpoints": [{"evaluation_sets": evaluation}],
        },
    }


def test_reporting_writes_json_markdown_and_latex(tmp_path) -> None:
    raw = tmp_path / "results" / "width_8" / "class_ordered" / "raw"
    raw.mkdir(parents=True)
    (raw / "dense_exact__seed_3.json").write_text(
        json.dumps(_run("dense_exact", 0.9)), encoding="utf-8"
    )
    (raw / "sparse_nystrom__seed_3.json").write_text(
        json.dumps(
            _run(
                "sparse_nystrom",
                0.895,
                representation_bytes=10,
                solver_bytes=80,
                total_bytes=90,
                event_microseconds=18.0,
            )
        ),
        encoding="utf-8",
    )
    (raw / "sparse_exact__seed_3.json").write_text(
        json.dumps(
            _run(
                "sparse_exact",
                0.9,
                representation_bytes=10,
                solver_bytes=200,
                total_bytes=210,
                event_microseconds=20.0,
            )
        ),
        encoding="utf-8",
    )

    summary = summarize_projection_results(tmp_path / "results")
    candidate = next(row for row in summary["rows"] if row["condition"] == "sparse_nystrom")
    assert candidate["mean_paired_final_accuracy_delta"] == pytest.approx(-0.005)
    assert candidate["median_full_event_microseconds"] == 18.0
    assert candidate["mean_paired_solver_fraction_delta"] == pytest.approx(-0.6)
    assert candidate["gate_findings"]["passed"] is True
    assert summary["matched_stream_validation_pass"] is True
    written = write_projection_summary(summary, tmp_path / "summary")
    assert set(written) == {"json", "md", "tex"}
    markdown = written["md"].read_text(encoding="utf-8")
    assert "sparse_nystrom" in markdown
    assert "PASS" in markdown

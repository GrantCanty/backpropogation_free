"""Paired aggregation and publication tables for projection-memory studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from continual_core.metrics import classification_plasticity_summary
from continual_core.results import write_json_result


def _final_accuracy(run: Mapping[str, Any]) -> float:
    return float(
        run["training"]["checkpoints"][-1]["evaluation_sets"]["all"]["accuracy"]
    )


def _bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(9_173)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = np.mean(values[indices], axis=1)
    return [float(item) for item in np.quantile(means, (0.025, 0.975))]


def _paired_relative_deltas(
    runs: Sequence[Mapping[str, Any]],
    references: Mapping[int, Mapping[str, Any]],
    getter: Any,
) -> np.ndarray:
    """Return paired candidate/reference minus one values."""

    values: list[float] = []
    for run in runs:
        reference = references[int(run["seed"])]
        denominator = float(getter(reference))
        if denominator:
            values.append(float(getter(run)) / denominator - 1.0)
    return np.asarray(values, dtype=np.float64)


def _mean_and_ci(values: np.ndarray) -> tuple[float | None, list[float] | None]:
    if not len(values):
        return None, None
    return float(np.mean(values)), _bootstrap_mean_ci(values)


def _load_runs(root: Path) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for path in sorted(root.glob("width_*/*/raw/*.json")):
        try:
            width = int(path.parents[2].name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        protocol = path.parents[1].name
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("completed") is not True:
            continue
        condition = str(run["condition"])
        grouped.setdefault((width, protocol, condition), []).append(run)
    if not grouped:
        raise ValueError(f"no completed raw projection results found under {root}")
    return grouped


def summarize_projection_results(
    root: str | Path, *, baseline: str = "dense_exact"
) -> dict[str, Any]:
    """Aggregate raw runs with paired seed deltas and CL/resource metrics."""

    source = Path(root).resolve()
    grouped = _load_runs(source)
    problem_hashes: dict[tuple[int, str, int], set[str]] = {}
    for (width, protocol, _), runs in grouped.items():
        for run in runs:
            problem_hashes.setdefault((width, protocol, int(run["seed"])), set()).add(
                str(run.get("matched_problem_sha256"))
            )
    stream_violations = [
        {"width": width, "protocol": protocol, "seed": seed, "hashes": sorted(hashes)}
        for (width, protocol, seed), hashes in problem_hashes.items()
        if len(hashes) != 1 or hashes == {"None"}
    ]
    rows: list[dict[str, Any]] = []
    for (width, protocol, condition), runs in sorted(grouped.items()):
        runs.sort(key=lambda item: int(item["seed"]))
        baseline_runs = grouped.get((width, protocol, baseline), [])
        baseline_by_seed = {int(run["seed"]): run for run in baseline_runs}
        paired = [run for run in runs if int(run["seed"]) in baseline_by_seed]
        deltas = np.asarray(
            [
                _final_accuracy(run)
                - _final_accuracy(baseline_by_seed[int(run["seed"])])
                for run in paired
            ],
            dtype=np.float64,
        )
        feature = str(runs[0].get("parameters", {}).get("feature", "dense"))
        matched_baseline = f"{feature}_exact"
        matched_runs = grouped.get((width, protocol, matched_baseline), [])
        matched_by_seed = {int(run["seed"]): run for run in matched_runs}
        matched_paired = [run for run in runs if int(run["seed"]) in matched_by_seed]
        matched_deltas = np.asarray(
            [
                _final_accuracy(run)
                - _final_accuracy(matched_by_seed[int(run["seed"])])
                for run in matched_paired
            ],
            dtype=np.float64,
        )
        readout = str(
            runs[0].get("parameters", {}).get(
                "readout", "nystrom" if condition.endswith("nystrom") else "exact"
            )
        )
        gate_reference = matched_baseline if readout == "nystrom" else baseline
        gate_reference_runs = grouped.get((width, protocol, gate_reference), [])
        gate_reference_by_seed = {
            int(run["seed"]): run for run in gate_reference_runs
        }
        gate_paired = [
            run for run in runs if int(run["seed"]) in gate_reference_by_seed
        ]
        total_memory_relative = _paired_relative_deltas(
            gate_paired,
            gate_reference_by_seed,
            lambda item: item["resources"]["total_persistent_bytes"],
        )
        solver_memory_relative = _paired_relative_deltas(
            gate_paired,
            gate_reference_by_seed,
            lambda item: item["resources"]["solver_bytes"],
        )
        latency_paired = [
            run
            for run in gate_paired
            if int(run["resources"].get("measurement_schema_version", 0)) >= 2
            and int(
                gate_reference_by_seed[int(run["seed"])]["resources"].get(
                    "measurement_schema_version", 0
                )
            ) >= 2
        ]
        latency_relative = _paired_relative_deltas(
            latency_paired,
            gate_reference_by_seed,
            lambda item: item["resources"]["end_to_end_event_latency"][
                "median_microseconds"
            ],
        )
        total_memory_delta, total_memory_ci = _mean_and_ci(total_memory_relative)
        solver_memory_delta, solver_memory_ci = _mean_and_ci(solver_memory_relative)
        latency_delta, latency_ci = _mean_and_ci(latency_relative)
        continual = [
            dict(
                run.get("continual_learning_metrics")
                or classification_plasticity_summary(run["training"])
            )
            for run in runs
        ]
        event_latencies = [
            float(run["resources"]["end_to_end_event_latency"]["median_microseconds"])
            for run in runs
            if int(run["resources"].get("measurement_schema_version", 0)) >= 2
        ]
        metric_mean = lambda name: float(
            np.mean([float(item.get(name, 0.0)) for item in continual])
        )
        row = {
                "width": width,
                "protocol": protocol,
                "condition": condition,
                "seeds": [int(run["seed"]) for run in runs],
                "run_count": len(runs),
                "paired_seed_count": len(paired),
                "mean_online_accuracy": float(
                    np.mean([run["training"]["online_accuracy"] for run in runs])
                ),
                "mean_final_locked_accuracy": float(
                    np.mean([_final_accuracy(run) for run in runs])
                ),
                "mean_paired_final_accuracy_delta": (
                    float(np.mean(deltas)) if len(deltas) else None
                ),
                "paired_final_accuracy_delta_bootstrap_95_ci": (
                    _bootstrap_mean_ci(deltas) if len(deltas) else None
                ),
                "matched_exact_baseline": matched_baseline,
                "mean_paired_final_accuracy_delta_vs_matched_exact": (
                    float(np.mean(matched_deltas)) if len(matched_deltas) else None
                ),
                "paired_final_accuracy_delta_vs_matched_exact_bootstrap_95_ci": (
                    _bootstrap_mean_ci(matched_deltas)
                    if len(matched_deltas) else None
                ),
                "gate_reference": gate_reference,
                "mean_paired_total_persistent_fraction_delta": total_memory_delta,
                "paired_total_persistent_fraction_delta_bootstrap_95_ci": total_memory_ci,
                "mean_paired_solver_fraction_delta": solver_memory_delta,
                "paired_solver_fraction_delta_bootstrap_95_ci": solver_memory_ci,
                "mean_paired_full_event_latency_fraction_delta": latency_delta,
                "paired_full_event_latency_fraction_delta_bootstrap_95_ci": latency_ci,
                "sample_efficiency_auc": metric_mean("sample_efficiency_auc"),
                "new_class_acquisition_accuracy": metric_mean(
                    "mean_new_class_acquisition_accuracy"
                ),
                "new_class_acquisition_gain": metric_mean(
                    "mean_new_class_acquisition_gain"
                ),
                "old_class_retention_delta": metric_mean(
                    "mean_old_class_retention_delta"
                ),
                "average_forgetting": metric_mean("average_forgetting"),
                "backward_transfer": metric_mean("backward_transfer"),
                "mean_relearning_gain": metric_mean("mean_relearning_gain"),
                "mean_segment_adaptation_delta": metric_mean(
                    "mean_segment_adaptation_delta"
                ),
                "mean_events_to_90pct_tail_accuracy": metric_mean(
                    "mean_events_to_90pct_tail_accuracy"
                ),
                "mean_representation_bytes": float(
                    np.mean([run["resources"]["representation_bytes"] for run in runs])
                ),
                "mean_solver_bytes": float(
                    np.mean([run["resources"]["solver_bytes"] for run in runs])
                ),
                "mean_total_persistent_bytes": float(
                    np.mean([run["resources"]["total_persistent_bytes"] for run in runs])
                ),
                "median_throughput_events_per_second": float(
                    np.median(
                        [run["resources"]["throughput_events_per_second"] for run in runs]
                    )
                ),
                "median_solver_update_microseconds": float(
                    np.median(
                        [
                            run["resources"]["solver_update_latency"]["median_microseconds"]
                            for run in runs
                        ]
                    )
                ),
                "median_full_event_microseconds": (
                    float(np.median(event_latencies))
                    if len(event_latencies) == len(runs)
                    else None
                ),
                "timing_schema_complete": len(event_latencies) == len(runs),
                "nonfinite_state_values": int(
                    sum(run["training"]["nonfinite_state_values"] for run in runs)
                ),
                "configuration_hashes": sorted(
                    {str(run["configuration_hash"]) for run in runs}
                ),
                "matched_problem_hashes": sorted(
                    {str(run.get("matched_problem_sha256")) for run in runs}
                ),
                "timing_session_ids": sorted(
                    {
                        str(run.get("runtime_environment", {}).get("timing_session_id"))
                        for run in runs
                    }
                ),
            }
        if condition == baseline:
            row["gate_findings"] = {
                "applicable": False,
                "reference": None,
                "passed": None,
                "note": "primary reference condition",
            }
        else:
            gate_accuracy = (
                matched_deltas if readout == "nystrom" else deltas
            )
            accuracy_ci = _bootstrap_mean_ci(gate_accuracy) if len(gate_accuracy) else None
            reference_timing_ids = {
                str(run.get("runtime_environment", {}).get("timing_session_id"))
                for run in gate_reference_runs
            }
            candidate_timing_ids = set(row["timing_session_ids"])
            timing_comparable = (
                len(candidate_timing_ids) == 1
                and candidate_timing_ids == reference_timing_ids
                and len(latency_paired) == len(gate_paired)
                and bool(gate_paired)
            )
            accuracy_pass = bool(accuracy_ci and accuracy_ci[0] >= -0.01)
            latency_pass = bool(
                timing_comparable
                and latency_delta is not None
                and latency_delta <= 0.0
            )
            required = [accuracy_pass, latency_pass]
            findings: dict[str, Any] = {
                "applicable": True,
                "reference": gate_reference,
                "accuracy_loss_limit": -0.01,
                "accuracy_delta_bootstrap_95_ci": accuracy_ci,
                "accuracy_pass": accuracy_pass,
                "timing_session_comparable": timing_comparable,
                "event_latency_fraction_delta": latency_delta,
                "event_latency_pass": latency_pass,
            }
            if readout == "nystrom":
                solver_pass = bool(
                    solver_memory_delta is not None
                    and solver_memory_delta <= -0.50
                )
                reference_solver_share = float(
                    np.mean(
                        [
                            run["resources"]["solver_bytes"]
                            / run["resources"]["total_persistent_bytes"]
                            for run in gate_reference_runs
                        ]
                    )
                ) if gate_reference_runs else 0.0
                total_gate_applicable = reference_solver_share >= 0.25
                total_pass = (
                    bool(
                        total_memory_delta is not None
                        and total_memory_delta <= -0.25
                    )
                    if total_gate_applicable
                    else None
                )
                findings.update(
                    {
                        "solver_memory_reduction_required": 0.50,
                        "solver_memory_fraction_delta": solver_memory_delta,
                        "solver_memory_pass": solver_pass,
                        "total_memory_gate_applicable": total_gate_applicable,
                        "total_memory_reduction_required": 0.25,
                        "total_memory_fraction_delta": total_memory_delta,
                        "total_memory_pass": total_pass,
                    }
                )
                required.append(solver_pass)
                if total_pass is not None:
                    required.append(total_pass)
            else:
                total_pass = bool(
                    total_memory_delta is not None
                    and total_memory_delta <= -0.25
                )
                findings.update(
                    {
                        "total_memory_reduction_required": 0.25,
                        "total_memory_fraction_delta": total_memory_delta,
                        "total_memory_pass": total_pass,
                    }
                )
                required.append(total_pass)
            findings["passed"] = all(required)
            row["gate_findings"] = findings
        rows.append(row)
    aggregate_findings: list[dict[str, Any]] = []
    keys = sorted({(row["width"], row["condition"]) for row in rows})
    for width, condition in keys:
        condition_rows = [
            row
            for row in rows
            if row["width"] == width and row["condition"] == condition
        ]
        applicable = [
            row for row in condition_rows if row["gate_findings"]["applicable"]
        ]
        aggregate_findings.append(
            {
                "width": width,
                "condition": condition,
                "protocols": [row["protocol"] for row in condition_rows],
                "passed_all_protocols": (
                    all(row["gate_findings"]["passed"] for row in applicable)
                    if applicable else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "experiment": "projection_memory_final_summary",
        "source_directory": str(source),
        "baseline": baseline,
        "rows": rows,
        "gate_findings": aggregate_findings,
        "matched_stream_validation_pass": not stream_violations,
        "matched_stream_violations": stream_violations,
        "timing_note": (
            "Full-event latency is reported only when every contributing run "
            "uses measurement schema version 2 or later."
        ),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Projection and Memory Final Summary",
        "",
        "## Gate findings",
        "",
        "| Width | Condition | Protocols | Passed all |",
        "|---:|---|---|---|",
    ]
    for finding in summary["gate_findings"]:
        passed = finding["passed_all_protocols"]
        label = "reference" if passed is None else ("PASS" if passed else "FAIL")
        lines.append(
            f"| {finding['width']} | {finding['condition']} | "
            f"{', '.join(finding['protocols'])} | {label} |"
        )
    lines.append("")
    for protocol in sorted({row["protocol"] for row in summary["rows"]}):
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Width | Condition | Final acc. | Paired delta (95% CI) | "
                "Forgetting | BWT | Total MB | Events/s | Event µs |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in summary["rows"] if item["protocol"] == protocol]:
            delta = row["mean_paired_final_accuracy_delta"]
            ci = row["paired_final_accuracy_delta_bootstrap_95_ci"]
            paired = "—" if delta is None else f"{delta:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
            event = row["median_full_event_microseconds"]
            lines.append(
                f"| {row['width']} | {row['condition']} | "
                f"{row['mean_final_locked_accuracy']:.4f} | {paired} | "
                f"{row['average_forgetting']:.4f} | {row['backward_transfer']:.4f} | "
                f"{row['mean_total_persistent_bytes'] / 1_000_000:.3f} | "
                f"{row['median_throughput_events_per_second']:.1f} | "
                f"{'—' if event is None else f'{event:.1f}'} |"
            )
        lines.append("")
        lines.extend(
            [
                "### Plasticity details",
                "",
                "| Width | Condition | Sample AUC | Acquisition | Retention Δ | "
                "Relearning Δ | Adaptation Δ | Recovery events |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in [item for item in summary["rows"] if item["protocol"] == protocol]:
            lines.append(
                f"| {row['width']} | {row['condition']} | "
                f"{row['sample_efficiency_auc']:.4f} | "
                f"{row['new_class_acquisition_accuracy']:.4f} | "
                f"{row['old_class_retention_delta']:+.4f} | "
                f"{row['mean_relearning_gain']:+.4f} | "
                f"{row['mean_segment_adaptation_delta']:+.4f} | "
                f"{row['mean_events_to_90pct_tail_accuracy']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _latex(summary: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{rrlrrrrr}",
        r"\toprule",
        r"Width & Protocol & Condition & Final acc. & $\Delta$ acc. & Forgetting & MB & Events/s \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        delta = row["mean_paired_final_accuracy_delta"]
        protocol = str(row["protocol"]).replace("_", r"\_")
        condition = str(row["condition"]).replace("_", r"\_")
        lines.append(
            f"{row['width']} & {protocol} & {condition} & "
            f"{row['mean_final_locked_accuracy']:.4f} & "
            f"{'--' if delta is None else f'{delta:+.4f}'} & "
            f"{row['average_forgetting']:.4f} & "
            f"{row['mean_total_persistent_bytes'] / 1_000_000:.3f} & "
            f"{row['median_throughput_events_per_second']:.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tabular}{rrlrrrrr}",
            r"\toprule",
            r"Width & Protocol & Condition & AUC & Acquisition & Retention $\Delta$ & Relearning $\Delta$ & Recovery \\",
            r"\midrule",
        ]
    )
    for row in summary["rows"]:
        protocol = str(row["protocol"]).replace("_", r"\_")
        condition = str(row["condition"]).replace("_", r"\_")
        lines.append(
            f"{row['width']} & {protocol} & {condition} & "
            f"{row['sample_efficiency_auc']:.4f} & "
            f"{row['new_class_acquisition_accuracy']:.4f} & "
            f"{row['old_class_retention_delta']:+.4f} & "
            f"{row['mean_relearning_gain']:+.4f} & "
            f"{row['mean_events_to_90pct_tail_accuracy']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_projection_summary(
    summary: Mapping[str, Any], output: str | Path,
    formats: Sequence[str] = ("json", "md", "tex"),
) -> dict[str, Path]:
    """Write requested formats from one canonical summary mapping."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    unknown = set(formats) - {"json", "md", "tex"}
    if unknown:
        raise ValueError("unknown summary formats: " + ", ".join(sorted(unknown)))
    written: dict[str, Path] = {}
    if "json" in formats:
        written["json"] = write_json_result(summary, destination / "summary.json")
    if "md" in formats:
        written["md"] = destination / "SUMMARY.md"
        written["md"].write_text(_markdown(summary), encoding="utf-8")
    if "tex" in formats:
        written["tex"] = destination / "summary_table.tex"
        written["tex"].write_text(_latex(summary), encoding="utf-8")
    return written

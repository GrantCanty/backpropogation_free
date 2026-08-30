#!/usr/bin/env python3
"""Run confirmatory, cross-dataset, and feature-width OS-ELM studies."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

from continual_core.results import write_json_result
from continual_core.datasets.classification import load_classification_split
from experiments.solver_comparison import (
    SolverComparisonConfig,
    combine_fixed_solver_runs,
    run_fixed_solver_study,
    write_solver_artifacts,
)


CORE_METHODS = (
    "exact_rls",
    "fd_ridge",
    "memory_matched_exact_rls",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPOSITORY_ROOT / "configs" / "solver_campaign_reference.json"


def _seed_set(count: int, *, master_seed: int, excluded: set[int]) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("seed counts must be positive")
    rng = np.random.default_rng(master_seed)
    result: list[int] = []
    while len(result) < count:
        candidate = int(rng.integers(1, 2**31 - 1))
        if candidate not in excluded and candidate not in result:
            result.append(candidate)
    return tuple(result)


def _reference(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"frozen reference configuration not found at {path}; provide "
            "--reference-results with an existing comparison or reference file"
        ) from error
    selected = document.get("selected_hyperparameters")
    config = document.get("config")
    if not isinstance(selected, dict) or not isinstance(config, dict):
        raise ValueError("reference result lacks config or selected hyperparameters")
    for required in ("exact_rls", "fd_ridge"):
        if required not in selected:
            raise ValueError(f"reference result lacks {required} parameters")
    return config, selected


def _equivalence_summary(
    result: Mapping[str, Any], *, margin: float, bootstrap_samples: int = 10_000
) -> dict[str, Any]:
    protocols = result["paired_vs_exact_rls"]["memory_matched_exact_rls"]
    preferred = "class_ordered" if "class_ordered" in protocols else next(iter(protocols))
    differences = np.asarray(
        protocols[preferred]["final_locked_accuracy"]["values"],
        dtype=np.float64,
    )
    rng = np.random.default_rng(9_173)
    indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
    bootstrap_means = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "comparison": "memory_matched_exact_rls_minus_exact_rls",
        "protocol_used": preferred,
        "seed_count": len(differences),
        "mean_paired_delta": float(np.mean(differences)),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "predeclared_equivalence_margin": [-margin, margin],
        "sufficient_seed_count": len(differences) >= 20,
        "equivalent_within_margin": (
            bool(lower > -margin and upper < margin)
            if len(differences) >= 20
            else None
        ),
        "bootstrap_samples": bootstrap_samples,
    }


def _run_stage(
    config: SolverComparisonConfig,
    selected: Mapping[str, Mapping[str, Any]],
    destination: Path,
    *,
    jobs: int,
) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    actual_jobs = jobs
    if jobs == 1 or len(config.heldout_seeds) == 1:
        result = run_fixed_solver_study(config, selected, methods=CORE_METHODS)
        actual_jobs = 1
    else:
        if config.dataset == "fashion_mnist":
            load_classification_split(
                config.dataset,
                test_per_class=config.test_per_class,
                seed=config.heldout_seeds[0],
                allow_download=config.allow_download,
                cache_directory=config.dataset_cache_directory,
            )
        payloads = [
            (replace(config, heldout_seeds=(seed,)), dict(selected))
            for seed in config.heldout_seeds
        ]
        try:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                partial_results = list(executor.map(_run_seed, payloads))
        except (OSError, PermissionError) as error:
            warnings.warn(
                f"process parallelism unavailable ({error}); running sequentially",
                RuntimeWarning,
                stacklevel=2,
            )
            result = run_fixed_solver_study(
                config, selected, methods=CORE_METHODS
            )
            actual_jobs = 1
        else:
            heldout_runs = [
                run
                for partial in partial_results
                for run in partial["heldout_runs"]
            ]
            result = combine_fixed_solver_runs(
                config, selected, heldout_runs, methods=CORE_METHODS
            )
    result["protocol"]["parallel_jobs_requested"] = jobs
    result["protocol"]["parallel_jobs_actual"] = actual_jobs
    result["protocol"]["timing_under_contention"] = actual_jobs > 1
    write_solver_artifacts(result, destination)
    return result


def _run_seed(
    payload: tuple[SolverComparisonConfig, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    config, selected = payload
    return run_fixed_solver_study(config, selected, methods=CORE_METHODS)


def _dataset_config(args: argparse.Namespace, seeds: tuple[int, ...]) -> SolverComparisonConfig:
    return SolverComparisonConfig(
        dataset=args.dataset,
        dataset_path=str(args.dataset_path) if args.dataset_path else None,
        allow_download=args.allow_download,
        dataset_cache_directory=(
            str(args.dataset_cache) if args.dataset_cache else None
        ),
        hidden_size=args.hidden_size,
        test_per_class=args.test_per_class,
        augmentation_copies=args.augmentation_copies,
        augmentation_max_shift=args.augmentation_max_shift,
        augmentation_noise_std=args.augmentation_noise_std,
        development_seeds=(args.master_seed,),
        heldout_seeds=seeds,
        development_events_per_segment=1,
        heldout_events_per_segment=args.train_events_per_segment,
        regularization_grid=(1.0,),
        lms_learning_rates=(0.1,),
        block_sizes=(8,),
        rpls_components=(8,),
        sketch_ranks=(max(1, min(args.hidden_size - 1, args.hidden_size // 4)),),
    )


def _scaled_parameters(
    selected: Mapping[str, Mapping[str, Any]],
    *,
    width: int,
    reference_width: int,
) -> dict[str, dict[str, Any]]:
    result = {name: dict(values) for name, values in selected.items()}
    reference_rank = int(result["fd_ridge"]["sketch_rank"])
    scaled_rank = max(1, round(reference_rank * width / reference_width))
    result["fd_ridge"]["sketch_rank"] = min(width, scaled_rank)
    return result


def _render_campaign(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# OS-ELM confirmatory campaign",
        "",
        f"Reference result: `{manifest['reference_results']}`",
        f"Cross-dataset source: `{manifest.get('dataset')}`",
        "",
    ]
    for stage, details in manifest["stages"].items():
        lines.extend([f"## {stage}", ""])
        if stage == "width_scaling":
            for width, item in details.items():
                quality = item["findings"]["quality_scores"]
                lines.append(
                    f"- Width {width}: exact={quality['exact_rls']:.4f}, "
                    f"smaller={quality['memory_matched_exact_rls']:.4f}, "
                    f"sketch={quality['fd_ridge']:.4f}."
                )
        else:
            quality = details["findings"]["quality_scores"]
            lines.append(
                f"- exact={quality['exact_rls']:.4f}, "
                f"smaller={quality['memory_matched_exact_rls']:.4f}, "
                f"sketch={quality['fd_ridge']:.4f}."
            )
            if "equivalence" in details:
                equivalence = details["equivalence"]
                lines.append(
                    f"- Smaller-vs-full paired delta={equivalence['mean_paired_delta']:+.4f}; "
                    f"95% bootstrap CI={equivalence['bootstrap_95_ci']}; "
                    f"equivalent={equivalence['equivalent_within_margin']}."
                )
        lines.append("")
    lines.extend(
        [
            "Every stage freezes hyperparameters before its held-out runs. "
            "Dataset downloads require an explicit command-line opt-in.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("confirmatory", "cross-dataset", "width-scaling", "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset",
        choices=("digits", "fashion_mnist", "npz"),
        help="dataset for cross-dataset and width-scaling stages",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=DEFAULT_REFERENCE,
    )
    parser.add_argument("--output", type=Path, default=Path("results/solver_campaign"))
    parser.add_argument("--confirm-seeds", type=int, default=50)
    parser.add_argument("--cross-seeds", type=int, default=20)
    parser.add_argument("--width-seeds", type=int, default=20)
    parser.add_argument("--widths", default="16,32,64,128")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--train-events-per-segment", type=int, default=1000)
    parser.add_argument("--augmentation-copies", type=int, default=0)
    parser.add_argument("--augmentation-max-shift", type=int, default=1)
    parser.add_argument("--augmentation-noise-std", type=float, default=0.03)
    parser.add_argument("--equivalence-margin", type=float, default=0.01)
    parser.add_argument("--master-seed", type=int, default=202_608_21)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel seed processes; use 1 when latency measurements matter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage in ("cross-dataset", "width-scaling", "all"):
        if args.dataset is None:
            raise SystemExit("--dataset is required for this stage")
        if args.dataset == "npz" and args.dataset_path is None:
            raise SystemExit("--dataset-path is required with --dataset npz")
    reference_config, selected = _reference(args.reference_results)
    excluded = set(reference_config.get("development_seeds", ()))
    excluded.update(reference_config.get("heldout_seeds", ()))
    excluded.add(args.master_seed)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "reference_results": str(args.reference_results),
        "dataset": args.dataset,
        "download_allowed": args.allow_download,
        "parallel_jobs": args.jobs,
        "stages": {},
    }

    if args.stage in ("confirmatory", "all"):
        seeds = _seed_set(
            args.confirm_seeds, master_seed=args.master_seed, excluded=excluded
        )
        excluded.update(seeds)
        confirmation = SolverComparisonConfig(
            dataset="digits",
            hidden_size=int(reference_config["hidden_size"]),
            test_per_class=int(reference_config["test_per_class"]),
            augmentation_copies=int(reference_config["augmentation_copies"]),
            augmentation_max_shift=int(reference_config["augmentation_max_shift"]),
            augmentation_noise_std=float(reference_config["augmentation_noise_std"]),
            development_seeds=(args.master_seed,),
            heldout_seeds=seeds,
            development_events_per_segment=1,
            heldout_events_per_segment=None,
            regularization_grid=(1.0,),
            lms_learning_rates=(0.1,),
            block_sizes=(8,),
            rpls_components=(8,),
            sketch_ranks=(int(selected["fd_ridge"]["sketch_rank"]),),
        )
        result = _run_stage(
            confirmation,
            selected,
            args.output / "confirmatory_digits",
            jobs=args.jobs,
        )
        manifest["stages"]["confirmatory"] = {
            "seeds": list(seeds),
            "execution": result["protocol"],
            "findings": result["findings"],
            "equivalence": _equivalence_summary(
                result, margin=args.equivalence_margin
            ),
        }

    if args.stage in ("cross-dataset", "all"):
        seeds = _seed_set(
            args.cross_seeds, master_seed=args.master_seed + 1, excluded=excluded
        )
        excluded.update(seeds)
        config = _dataset_config(args, seeds)
        result = _run_stage(
            config,
            selected,
            args.output / f"cross_{args.dataset}",
            jobs=args.jobs,
        )
        manifest["stages"]["cross_dataset"] = {
            "seeds": list(seeds),
            "execution": result["protocol"],
            "findings": result["findings"],
            "equivalence": _equivalence_summary(
                result, margin=args.equivalence_margin
            ),
        }

    if args.stage in ("width-scaling", "all"):
        widths = tuple(int(value) for value in args.widths.split(","))
        if not widths or any(width < 4 for width in widths):
            raise SystemExit("--widths must contain comma-separated integers >= 4")
        seeds = _seed_set(
            args.width_seeds, master_seed=args.master_seed + 2, excluded=excluded
        )
        base = _dataset_config(args, seeds)
        reference_width = int(reference_config["hidden_size"])
        width_results: dict[str, Any] = {}
        for width in widths:
            config = replace(base, hidden_size=width)
            parameters = _scaled_parameters(
                selected, width=width, reference_width=reference_width
            )
            result = _run_stage(
                config,
                parameters,
                args.output / "width_scaling" / f"width_{width}",
                jobs=args.jobs,
            )
            width_results[str(width)] = {
                "seeds": list(seeds),
                "execution": result["protocol"],
                "sketch_rank": parameters["fd_ridge"]["sketch_rank"],
                "findings": result["findings"],
                "equivalence": _equivalence_summary(
                    result, margin=args.equivalence_margin
                ),
            }
        manifest["stages"]["width_scaling"] = width_results

    args.output.mkdir(parents=True, exist_ok=True)
    write_json_result(manifest, args.output / "campaign.json")
    (args.output / "CAMPAIGN.md").write_text(
        _render_campaign(manifest), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

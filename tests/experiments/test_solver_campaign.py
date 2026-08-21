import json

from scripts.run_solver_campaign import main


def test_width_campaign_accepts_dataset_argument_and_writes_manifest(
    tmp_path,
) -> None:
    reference = {
        "config": {
            "hidden_size": 8,
            "test_per_class": 2,
            "augmentation_copies": 0,
            "augmentation_max_shift": 1,
            "augmentation_noise_std": 0.0,
            "development_seeds": [1],
            "heldout_seeds": [2],
        },
        "selected_hyperparameters": {
            "exact_rls": {"regularization": 1.0},
            "fd_ridge": {"regularization": 1.0, "sketch_rank": 2},
        },
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    output = tmp_path / "campaign"
    assert (
        main(
            [
                "--stage",
                "width-scaling",
                "--dataset",
                "digits",
                "--reference-results",
                str(reference_path),
                "--output",
                str(output),
                "--width-seeds",
                "1",
                "--widths",
                "4",
                "--test-per-class",
                "2",
                "--train-events-per-segment",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "campaign.json").read_text())
    assert manifest["dataset"] == "digits"
    assert set(manifest["stages"]["width_scaling"]) == {"4"}
    assert (
        output / "width_scaling" / "width_4" / "comparison.json"
    ).is_file()

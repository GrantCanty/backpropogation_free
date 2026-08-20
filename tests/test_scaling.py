from no_backprop.scaling import (
    BlankImageScalingConfig,
    FeatureWidthScalingConfig,
    run_blank_image_scaling,
    run_feature_width_scaling,
)


def test_blank_image_scaling_is_lazy_and_bounded() -> None:
    result = run_blank_image_scaling(
        BlankImageScalingConfig(
            sample_counts=(12,),
            image_sizes=(4,),
            hidden_size=6,
            classes=2,
            block_size=3,
            kinds=("rls", "diagonal_rls", "prototype"),
        )
    )
    assert all(run["bounded_state"] for run in result["runs"])
    assert all(run["dataset_storage_bytes"] == 4 * 4 * 8 for run in result["runs"])


def test_feature_width_scaling_reports_measured_and_projected_state() -> None:
    result = run_feature_width_scaling(
        FeatureWidthScalingConfig(
            feature_widths=(5, 9),
            projected_widths=(5, 9, 17),
            updates=12,
            classes=2,
            block_size=4,
            kinds=("rls", "diagonal_rls", "block_rls"),
        )
    )
    assert len(result["measured"]) == 6
    assert len(result["projected"]) == 9
    exact = {
        item["feature_width"]: item["projected_state_bytes"]
        for item in result["projected"]
        if item["model"] == "rls"
    }
    diagonal = {
        item["feature_width"]: item["projected_state_bytes"]
        for item in result["projected"]
        if item["model"] == "diagonal_rls"
    }
    assert exact[17] > diagonal[17]

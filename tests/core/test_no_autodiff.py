from pathlib import Path


def test_core_has_no_automatic_differentiation_dependencies() -> None:
    root = Path(__file__).parents[2] / "src" / "continual_core"
    forbidden = ("import torch", "from torch", ".backward(", "autograd")
    violations = []
    for path in root.glob("*.py"):
        text = path.read_text().lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.name}: {token}")
    assert not violations, violations


def test_core_does_not_import_comparison_baselines() -> None:
    root = Path(__file__).parents[2] / "src" / "continual_core"
    violations = [
        path.name
        for path in root.glob("*.py")
        if "baselines" in path.read_text().lower()
    ]
    assert not violations, violations

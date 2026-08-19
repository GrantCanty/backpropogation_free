from pathlib import Path


def test_core_has_no_automatic_differentiation_dependencies() -> None:
    root = Path(__file__).parents[1] / "src" / "no_backprop"
    forbidden = ("import torch", "from torch", ".backward(", "autograd")
    violations = []
    for path in root.glob("*.py"):
        text = path.read_text().lower()
        for token in forbidden:
            if token in text:
                violations.append(f"{path.name}: {token}")
    assert not violations, violations

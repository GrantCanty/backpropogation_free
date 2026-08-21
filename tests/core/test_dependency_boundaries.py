"""Architectural tests that keep disposable methods out of shared code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).parents[2] / "src"


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    ("package", "forbidden"),
    (
        ("continual_core", {"baselines", "methods", "experiments", "no_backprop"}),
        ("baselines", {"methods", "experiments", "no_backprop"}),
        ("methods", {"baselines", "experiments", "no_backprop"}),
    ),
)
def test_source_dependency_direction(package: str, forbidden: set[str]) -> None:
    violations: list[str] = []
    for path in sorted((SOURCE / package).rglob("*.py")):
        invalid = imported_roots(path) & forbidden
        if invalid:
            violations.append(
                f"{path.relative_to(SOURCE)} imports {', '.join(sorted(invalid))}"
            )
    assert not violations, "\n".join(violations)


def test_retired_cpam_is_not_in_the_active_source_tree() -> None:
    assert not (SOURCE / "methods" / "cpam").exists()
    violations = []
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "methods.cpam" in text or "archives.cpam" in text:
            violations.append(str(path.relative_to(SOURCE)))
    assert not violations, "active CPAM references: " + ", ".join(violations)

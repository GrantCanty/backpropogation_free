from no_backprop import __version__
from no_backprop.cli import main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_starts() -> None:
    assert main([]) == 0
